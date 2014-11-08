(ns linear-regression.core-test
  (:require [clojure.test :refer :all]
            [linear-regression.core :refer :all]))

(defmacro hundredths [x]
  `(float (/ (Math/round (* ~x 100)) 100)))

(deftest height-test
  (testing "Testing age vs height regression"
    (let [training-inputs (map #(Double/valueOf %) (clojure.string/split (slurp "/home/seditiosus/clojure/linear-regression/ex2x.dat") #"\n"))
          training-outputs (map #(Double/valueOf %) (clojure.string/split (slurp "/home/seditiosus/clojure/linear-regression/ex2y.dat") #"\n"))
          [theta0 theta1] (univar-linear-regression 0.07 [0 0]
                                                    training-inputs
                                                    training-outputs)
          y1 (hundredths (hypothesis theta0 theta1 3.5))
          y2 (hundredths (hypothesis theta0 theta1 7))]
      (is (= y1 (float 0.97)))
      (is (= y2 (float 1.2))))))

;(deftest crickets-test
;  (let [data (drop 2 (clojure.string/split (slurp "crickets.csv") #"[\n,]"))
;        train-in (map #(Double/valueOf %) (take-nth 2 data))
;        train-out (map #(Double/valueOf %) (take-nth 2 (rest data)))]
;    (println (univar-linear-regression alpha thetas train-in train-out))))
