(ns usgoc.export-results
  (:require [babashka.pods :as pods]
            [clojure.math :as math]
            [clojure.string :as str]
            [clojure.java.io :as io]
            [clojure.data.csv :as csv]))

(pods/load-pod 'org.babashka/go-sqlite3 "0.1.0")
(require '[pod.babashka.go-sqlite3 :as sqlite])

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
        base ["accuracy" "label1_accuracy" "label2_accuracy"]
        stat [:mean :std]
        :let [split-base (str/join "_" (filter some? [split base]))]]
    [(keyword (str split-base "_" (name stat)))
     (comp stat (keyword split-base) :metrics)]))


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

(defn get-runs
  [& {:keys [experiment-id include-metrics? aggregate-children?]
      :or {experiment-id (get-experiment-id)
           include-metrics? false
           aggregate-children? false}}]
  (let [runs (sqlite/query db-path ["SELECT * FROM runs WHERE experiment_id = ?" experiment-id])
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
    parent-runs))

(defn write-csv
  ([path cols maps]
   (write-csv path cols (map (apply juxt cols)) maps))
  ([path cols xform maps]
   (with-open [w (io/writer path)]
     (let [data (into [(mapv name cols)]
                      xform
                      maps)]
       (csv/write-csv w data)))))

(defn write-aggregate-runs
  []
  (let [runs (get-runs :include-metrics? true :aggregate-children? true)
        model-kws [:model :limit_id :convert_mode :dataset]
        metric-kws (map first metric-accessors)]
    (println model-kws metric-kws)
    (write-csv "results/results.csv"
               (into model-kws metric-kws)
               (map (fn [run]
                      (into ((apply juxt model-kws) run)
                            (comp (map second) (map #(% run)))
                            metric-accessors)))
               runs)))

(comment
  (let [runs (time (get-runs :include-metrics? true :aggregate-children? true))
        fr (first runs)]
    (map (second (nth metric-accessors 3)) runs))
  (time (get-metrics "379f8efef8d2421fba9977477de35ceb"))
  (write-aggregate-runs)
  )
