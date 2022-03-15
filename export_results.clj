(ns usgoc.export-results
  (:require [babashka.deps :as deps]
            [babashka.pods :as pods]
            [clojure.math :as math]
            [clojure.string :as str]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(deps/add-deps '{:deps {camel-snake-kebab/camel-snake-kebab {:mvn/version "0.4.2"}}})
(pods/load-pod 'org.babashka/go-sqlite3 "0.1.0")
(require '[pod.babashka.go-sqlite3 :as sqlite]
         '[camel-snake-kebab.core :as csk])

(def db-path "./mlflow.db")
(def results-path "./results")
(def experiment-name "usgo_v1")
#_(def top-metrics-query
    "SELECT m1.run_uuid as run_uuid, m1.key as key, m1.value as value
   FROM metrics m1, (
     SELECT run_uuid, CAST(value AS INT) AS max_step FROM metrics
     WHERE run_uuid = ? AND key = \"restored_epoch\"
   ) m2
   WHERE m1.run_uuid = m2.run_uuid AND (m1.step = m2.max_step OR m1.step = -1)")
; Simple metrics query can be used if database only contains the last metric values:
(def top-metrics-query "SELECT * FROM metrics WHERE run_uuid = ?")
(def complete-deterministic-run-child-count 10)
(def complete-run-child-count 30)

(def model-name-replacements {"WL2GNN" "2-WL-GNN"})
(def limit-id-replacements {"v127_d127_f127_p127" "all"
                            "v127_d0_f0_p0_no_tf_fb_ob_ou" "only vars"
                            "v0_d127_f0_p0_no_v_vt_fb_ob_ou" "only types"
                            "v0_d0_f127_p0_no_v_vt_tf" "only funcs"
                            "v0_d0_f0_p127_no_v_vt_tf_fb_ob_ou" "only pkgs"
                            "v0_d0_f0_p0_no_v_vt_tf_fb_ob_ou" "none"})
(def model-order (zipmap ["Majority" "MLP" "DeepSets" "GIN" "WL2GNN"] (range)))
(def limit-id-order (zipmap ["v127_d127_f127_p127"
                             "v127_d0_f0_p0_no_tf_fb_ob_ou"
                             "v0_d127_f0_p0_no_v_vt_fb_ob_ou"
                             "v0_d0_f127_p0_no_v_vt_tf"
                             "v0_d0_f0_p127_no_v_vt_tf_fb_ob_ou"
                             "v0_d0_f0_p0_no_v_vt_tf_fb_ob_ou"]
                            (range)))
; One-sided t-CDF values:
(def t-cdf {9 {0.1 1.383, 0.05 1.833, 0.01 2.821}})

(def metric-accessors
  (for [split [nil "val" "test"]
        base ["loss" "label1_loss" "label2_loss"
              "accuracy" "label1_accuracy" "label2_accuracy"
              "accuracy_conf_0.1" "label1_accuracy_conf_0.1" "label2_accuracy_conf_0.1"
              "label1_mean_size_conf_0.1" "label2_mean_size_conf_0.1"]
        stat [:mean :std]
        :let [split-base (str/join "_" (filter some? [split base]))
              split-base-kw (keyword split-base)
              metric-desc {:name (keyword (str split-base "_" (name stat)))
                           :accessor (comp stat split-base-kw :metrics)}]]
    (if (= stat :mean)
      (assoc metric-desc
             :best-model-name (keyword (str split-base "_best_model"))
             :best-model-accessor (comp :best-model? split-base-kw :metrics)
             :best-limit-id-name (keyword (str split-base "_best_limit_id"))
             :best-limit-id-accessor (comp :best-limit-id? split-base-kw :metrics)
             :worst-model-name (keyword (str split-base "_worst_model"))
             :worst-model-accessor (comp :worst-model? split-base-kw :metrics)
             :worst-limit-id-name (keyword (str split-base "_worst_limit_id"))
             :worst-limit-id-accessor (comp :worst-limit-id? split-base-kw :metrics))
      metric-desc)))

(defn get-default [m] #(get m % %))

(defn loss-metric? [metric]
  (not (str/includes? (name metric) "accuracy")))

(defn mean [vals] (/ (apply + vals) (count vals)))

(defn mean-var
  [vals & [vars]]
  (let [m (mean vals)
        v (cond
            (some? vars) (/ (apply + vars) (#(* % %) (count vars)))
            (= (count vals) 1) 0
            :else
            (/ (apply + (map (comp #(* % %) #(- % m))
                             vals))
               (dec (count vals))))]
    [m v]))

(defn stats
  [vals
   & {:keys [with-var? val-fn var-fn min-fn max-fn count-fn]
      :or {with-var? false
           val-fn first
           var-fn second
           min-fn nil
           max-fn nil
           count-fn nil}}]
  (if (empty? vals)
    {:vals [] :vars []
     :mean 0 :var 0 :std 0 :ste 0
     :min 0 :max 0 :outer-count 0 :count 0}
    (let [[inner-vals inner-vars :as mean-args]
          (if with-var?
            [(map val-fn vals) (map var-fn vals)]
            [vals])
          [vmean vvar] (apply mean-var mean-args)
          vstd (math/sqrt vvar)
          outer-count (count vals)
          vste (if with-var? vstd (/ vstd (math/sqrt outer-count)))
          vmin (apply min (if (nil? min-fn)
                            inner-vals
                            (map min-fn vals)))
          vmax (apply max (if (nil? max-fn)
                            inner-vals
                            (map max-fn vals)))]
      {:vals inner-vals
       :vars inner-vars
       :mean vmean
       :var vvar
       :std vstd
       :ste vste
       :min vmin
       :max vmax
       :outer-count outer-count
       :count (if (nil? count-fn)
                outer-count
                (transduce (map count-fn) + vals))})))

(defn get-experiment-id
  []
  (let [[{id :experiment_id}]
        (sqlite/query db-path ["SELECT * FROM experiments WHERE name = ?" experiment-name])]
    id))

(defn group-by-run
  [run-ids s]
  (->> s
       (filter (comp run-ids :run_uuid))
       (group-by :run_uuid)))

(defn kv-list->map
  [kv-list]
  (into {} (map (juxt (comp keyword :key) :value)) kv-list))

(defn get-metrics
  [run-id]
  (kv-list->map (sqlite/query db-path [top-metrics-query run-id])))

(defn aggregate-maps
  [maps & {:as opts}]
  (as-> maps $
    (filter some? $)
    (map #(update-vals % vector) $)
    (apply merge-with into $)
    (update-vals $ #(stats % opts))))

(defn aggregate-runs
  [run-seq]
  (let [folds (vals (group-by :fold run-seq))
        folds (map #(aggregate-maps (map :metrics %)) folds)]
    (aggregate-maps folds
                    :with-var? true
                    :val-fn :mean
                    :var-fn :ste
                    :min-fn :min :max-fn :max
                    :count-fn :count)))

(defn paired-test
  [level smaller? {vals1 :vals, vars1 :vars} {vals2 :vals, vars2 :vars}]
  (let [n (count vals1)]
    (if (zero? n)
      false
      (let [val-diffs (if smaller?
                        (map - vals2 vals1)
                        (map - vals1 vals2))
            var-diffs (map + vars1 vars2)
            [diff-mean diff-var] (mean-var val-diffs var-diffs)]
        (if (zero? diff-var)
          (<= diff-mean 0)
          (<= (/ diff-mean (math/sqrt diff-var))
              (get-in t-cdf [(dec n) level])))))))

(defn find-best-run-metrics
  ([run-seq level]
   (find-best-run-metrics run-seq level false))
  ([run-seq level invert]
   (if (= (count run-seq) 1)
     (let [[{:keys [metrics]}] run-seq]
       (zipmap metrics (repeat (constantly false))))
     (let [best-map
           (persistent!
            (reduce (fn [best-metrics run]
                      (let []
                        (reduce-kv (fn [best-metrics metric {:keys [mean] :as stats}]
                                     (let [min-metric? (loss-metric? metric)
                                           min-metric? (if invert (not min-metric?) min-metric?)
                                           pred (if min-metric? < >)
                                           {:keys [best-mean]} (best-metrics metric)]
                                       (if (or (nil? best-mean)
                                               (pred mean best-mean))
                                         (assoc! best-metrics metric
                                                 {:best-mean mean
                                                  :best-test
                                                  #(paired-test level min-metric? stats %)})
                                         best-metrics)))
                                   best-metrics
                                   (:metrics run))))
                    (transient {}) run-seq))]
       (update-vals best-map :best-test)))))

(defn grouped-best-run-metrics
  [run-seq group-fn level]
  (let [run-groups (group-by group-fn run-seq)
        best-tests (update-vals run-groups
                                #(find-best-run-metrics % level false))
        worst-tests (update-vals run-groups
                                 #(find-best-run-metrics % level true))]
    [best-tests worst-tests]))

(defn add-best-flags-to-metrics
  [metrics tests kw]
  (reduce-kv (fn [metrics metric best-test]
               (if (contains? metrics metric)
                 (assoc-in metrics [metric kw]
                           (-> metrics metric best-test))
                 metrics))
             metrics tests))

(defn get-raw-runs
  [& {:keys [experiment-id include-metrics?]
      :or {experiment-id (get-experiment-id)
           include-metrics? false}}]
  (let [runs (sqlite/query db-path ["SELECT * FROM runs WHERE experiment_id = ? ORDER BY name"
                                    experiment-id])
        run-ids (into #{} (map :run_uuid) runs)
        tags (group-by-run run-ids (sqlite/query db-path ["SELECT * FROM tags"]))
        params (group-by-run run-ids (sqlite/query db-path ["SELECT * FROM params"]))
        runs (map (fn [run]
                    (let [run-uuid (:run_uuid run)]
                      (merge (select-keys run
                                          [:run_uuid :name :status
                                           :start_time :end_time :artifact_uri])
                             (kv-list->map (tags run-uuid))
                             {:params (kv-list->map (params run-uuid))
                              :metrics (when include-metrics?
                                         (get-metrics run-uuid))})))
                  runs)]
    runs))

(defn group-runs
  [runs
   & {:keys [aggregate-children?]
      :or {aggregate-children? false}}]
  (let [runs (group-by :mlflow.parentRunId runs)
        parent-runs (map (fn [run]
                           (let [run (dissoc run :metrics)
                                 children (runs (:run_uuid run))
                                 run (assoc run :complete?
                                            (= (count children)
                                               (if (:deterministic run)
                                                 complete-deterministic-run-child-count
                                                 complete-run-child-count)))]
                             (if aggregate-children?
                               (assoc run :metrics
                                      (aggregate-runs children))
                               (assoc run :children children))))
                         (runs nil))]
    (println "Got" (count parent-runs) "parent runs.")
    (if aggregate-children?
      (let [level 0.1
            [best-model-tests worst-model-tests]
            (grouped-best-run-metrics parent-runs :limit_id level)
            [best-limit-id-tests worst-limit-id-tests]
            (grouped-best-run-metrics parent-runs :model level)]
        (map (fn [run]
               (update run :metrics
                       #(-> %
                            (add-best-flags-to-metrics (-> run :limit_id best-model-tests)
                                                       :best-model?)
                            (add-best-flags-to-metrics (-> run :limit_id worst-model-tests)
                                                       :worst-model?)
                            (add-best-flags-to-metrics (-> run :model best-limit-id-tests)
                                                       :best-limit-id?)
                            (add-best-flags-to-metrics (-> run :model worst-limit-id-tests)
                                                       :worst-limit-id?))))
             parent-runs))
      parent-runs)))

(defn get-runs
  [& {:as opts}]
  (let [runs (get-raw-runs opts)]
    (group-runs runs opts)))

(defn normalize-colname
  [colname]
  (-> colname
      name
      csk/->camelCase
      (str/replace #"0\.(\d+)"
                   (fn [[_ decs]] (apply str "p" (map #(char (+ 17 (int %))) decs))))
      (str/replace #"\d+" #(apply str (repeat (parse-long %) \I)))))

(defn write-csv
  ([path cols maps]
   (write-csv path cols (map (apply juxt cols)) maps))
  ([path cols xform maps]
   (with-open [w (io/writer path)]
     (let [data
           (into [(mapv normalize-colname cols)]
                 xform maps)]
       (csv/write-csv w data)))))

(defn write-aggregate-runs
  []
  (println "Fetching runs...")
  (let [runs (get-runs :include-metrics? true :aggregate-children? true)
        _ (println "Loaded" (count runs) "runs.")
        runs (filter :complete? runs)
        runs (sort-by (juxt (comp limit-id-order :limit_id)
                            (comp model-order :model))
                      runs)
        model-kws [:model :limit_id :convert_mode :dataset]
        metric-kws (map :name metric-accessors)
        best-model-kws (keep :best-model-name metric-accessors)
        best-limit-id-kws (keep :best-limit-id-name metric-accessors)
        worst-model-kws (keep :worst-model-name metric-accessors)
        worst-limit-id-kws (keep :worst-limit-id-name metric-accessors)
        columns (-> model-kws
                    (into metric-kws)
                    (into best-model-kws)
                    (into best-limit-id-kws)
                    (into worst-model-kws)
                    (into worst-limit-id-kws))]
    (println (str "Writing results CSV. "
                  (count runs) " rows, " (count columns) " columns."))
    (write-csv "results/results.csv"
               columns
               (map (fn [run]
                      (-> run
                          ((juxt (comp (get-default model-name-replacements)
                                       :model)
                                 (comp (get-default limit-id-replacements)
                                       :limit_id)
                                 :convert_mode :dataset))
                          (into (comp (map :accessor) (map #(% run)))
                                metric-accessors)
                          (into (comp (keep :best-model-accessor)
                                      (map #(% run)) (map #(if % 1 0)))
                                metric-accessors)
                          (into (comp (keep :best-limit-id-accessor)
                                      (map #(% run)) (map #(if % 1 0)))
                                metric-accessors)
                          (into (comp (keep :worst-model-accessor)
                                      (map #(% run)) (map #(if % 1 0)))
                                metric-accessors)
                          (into (comp (keep :worst-limit-id-accessor)
                                      (map #(% run)) (map #(if % 1 0)))
                                metric-accessors))))
               runs)))

(comment
  (def raw-runs (time (get-raw-runs :include-metrics? true)))
  (as-> raw-runs $ (group-by :status $) (get $ "FAILED") (first $))
  (let [runs (time (group-runs raw-runs :aggregate-children? true))
        fr (first (filter #(= (:model %) "Majority") runs))]
    #_(map (second (nth metric-accessors 3)) runs)
    #_(->> runs (remove :complete?) first)
    (-> fr ((juxt :model :limit_id :deterministic :complete?))))
  (time (get-metrics "379f8efef8d2421fba9977477de35ceb"))
  (with-redefs [get-raw-runs (constantly raw-runs)]
    (write-aggregate-runs))
  )
