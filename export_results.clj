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
(def top-metrics-query
  "SELECT m1.run_uuid as run_uuid, m1.key as key, m1.value as value
   FROM metrics m1, (
     SELECT run_uuid, CAST(value AS INT) AS max_step FROM metrics
     WHERE run_uuid = ? AND key = \"restored_epoch\"
   ) m2
   WHERE m1.run_uuid = m2.run_uuid AND (m1.step = m2.max_step OR m1.step = -1)")

(def metric-accessors
  (for [split [nil "val" "test"]
        base ["loss" "label1_loss" "label2_loss"
              "accuracy" "label1_accuracy" "label2_accuracy"]
        stat [:mean :std]
        :let [split-base (str/join "_" (filter some? [split base]))
              split-base-kw (keyword split-base)
              metric-desc {:name (keyword (str split-base "_" (name stat)))
                           :accessor (comp stat split-base-kw :metrics)}]]
    (if (= stat :mean)
      (assoc metric-desc
             :best-name (keyword (str split-base "_best"))
             :best-accessor (comp :best? split-base-kw :metrics))
      metric-desc)))

(defn loss-metric? [metric] (str/ends-with? (name metric) "loss"))

(defn mean [vals] (/ (apply + vals) (count vals)))

(defn mean-std
  [vals]
  (let [m (mean vals)]
    [m (math/sqrt
        (/ (apply + (map (comp #(* % %) #(- % m))
                         vals))
           (dec (count vals))))]))

(defn stats
  [vals]
  (let [[vmean vstd] (mean-std vals)
        vmin (apply min vals)
        vmax (apply max vals)]
    {:mean vmean
     :std vstd
     :min vmin
     :max vmax
     :count (count vals)}))

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
  [maps]
  (as-> maps $
    (filter some? $)
    (map #(update-vals % vector) $)
    (apply merge-with into $)
    (update-vals $ stats)))

(defn aggregate-runs
  [run-seq]
  (aggregate-maps (map :metrics run-seq)))

(defn find-best-run-metrics
  ([run-seq] (find-best-run-metrics run-seq :metrics identity))
  ([run-seq get-value] (find-best-run-metrics run-seq :metrics get-value))
  ([run-seq get-metrics get-value]
   (persistent!
    (reduce (fn [best-metrics run]
              (reduce-kv (fn [best-metrics metric value]
                           (let [value (get-value value)
                                 pred (if (loss-metric? metric) < >)
                                 best-value (best-metrics metric)]
                             (if (or (nil? best-value)
                                     (pred value best-value))
                               (assoc! best-metrics metric value)
                               best-metrics)))
                         best-metrics
                         (get-metrics run)))
            (transient {}) run-seq))))

(defn get-runs
  [& {:keys [experiment-id include-metrics? aggregate-children?]
      :or {experiment-id (get-experiment-id)
           include-metrics? false
           aggregate-children? false}}]
  (let [runs (sqlite/query db-path ["SELECT * FROM runs WHERE experiment_id = ? ORDER BY name"
                                    experiment-id])
        run-ids (into #{} (map :run_uuid) runs)
        tags (group-by-run run-ids (sqlite/query db-path ["SELECT * FROM tags"]))
        params (group-by-run run-ids (sqlite/query db-path ["SELECT * FROM params"]))
        runs (->> runs
                  (map (fn [run]
                         (let [run-uuid (:run_uuid run)]
                           (merge (select-keys run
                                               [:run_uuid :name :status
                                                :start_time :end_time :artifact_uri])
                                  (kv-list->map (tags run-uuid))
                                  {:params (kv-list->map (params run-uuid))
                                   :metrics (when include-metrics?
                                              (get-metrics run-uuid))}))))
                  (group-by :mlflow.parentRunId))
        parent-runs (map (fn [run]
                           (let [run (dissoc run :metrics)
                                 children (runs (:run_uuid run))]
                             (if aggregate-children?
                               (assoc run :metrics
                                      (aggregate-runs children))
                               (assoc run :children children))))
                         (runs nil))]
    (if aggregate-children?
      (let [best-means (find-best-run-metrics parent-runs :mean)]
        (map (fn [run]
               (let [metrics (:metrics run)
                     metrics (reduce-kv (fn [metrics metric best-mean]
                                          (assoc-in metrics [metric :best?]
                                                    (= (-> metrics metric :mean) best-mean)))
                                        metrics best-means)]
                 (assoc run :metrics metrics)))
             parent-runs))
      parent-runs)))

(defn write-csv
  ([path cols maps]
   (write-csv path cols (map (apply juxt cols)) maps))
  ([path cols xform maps]
   (with-open [w (io/writer path)]
     (let [data (into [(mapv (comp (fn [c]
                                     (str/replace c #"\d+"
                                                  #(apply str (repeat (parse-long %) \I))))
                                   csk/->camelCase name)
                             cols)]
                      xform
                      maps)]
       (csv/write-csv w data)))))

(defn write-aggregate-runs
  []
  (let [runs (get-runs :include-metrics? true :aggregate-children? true)
        model-kws [:model :limit_id :convert_mode :dataset]
        metric-kws (map :name metric-accessors)
        best-kws (keep :best-name metric-accessors)
        columns (-> model-kws (into metric-kws) (into best-kws))]
    (println (str "Writing results CSV. "
                  (count runs) " rows, " (count columns) " columns."))
    (write-csv "results/results.csv"
               columns
               (map (fn [run]
                      (-> ((apply juxt model-kws) run)
                          (into (comp (map :accessor) (map #(% run)))
                                metric-accessors)
                          (into (comp (keep :best-accessor)
                                      (map #(% run)) (map #(if % 1 0)))
                                metric-accessors))))
               runs)))

(comment
  (let [runs (time (get-runs :include-metrics? true :aggregate-children? true))
        fr (first runs)]
    (map (second (nth metric-accessors 3)) runs))
  (time (get-metrics "379f8efef8d2421fba9977477de35ceb"))
  (write-aggregate-runs)
  )
