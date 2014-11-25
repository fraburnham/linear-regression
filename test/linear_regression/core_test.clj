(ns linear-regression.core-test
  (:require [clojure.test :refer :all]
            [linear-regression.core :refer :all]))

(defmacro hundredths [x]
  `(float (/ (Math/round (* ~x 100)) 100)))

;TODO: write tests for each individual function to verify them
;make it a habit to write the test as you write the function
;bust out a few test cases to prove that the function works
;as they never have side effects it should guarantee the function
;will always work given valid inputs

(deftest univar-height-test
  (testing "Testing age vs height regression"
    (let [training-inputs (map #(Double/valueOf %) (clojure.string/split (slurp "/home/seditiosus/clojure/linear-regression/ex2x.dat") #"\n"))
          training-outputs (map #(Double/valueOf %) (clojure.string/split (slurp "/home/seditiosus/clojure/linear-regression/ex2y.dat") #"\n"))
          [theta0 theta1] (univar-linear-regression 0.07 [0 0]
                                                    training-inputs
                                                    training-outputs)
          y1 (hundredths (univar-hypothesis theta0 theta1 3.5))
          y2 (hundredths (univar-hypothesis theta0 theta1 7))]
      (is (= y1 (float 0.97)))
      (is (= y2 (float 1.2))))))

;run the height regression using the multivar stuff
(deftest height-test
  (testing "Testing age vs height regression"
    (let [training-inputs (map (partial cons 1)
                               (partition 1 (map #(Double/valueOf %)
                                                 (clojure.string/split
                                                   (slurp "/home/seditiosus/clojure/linear-regression/ex2x.dat")
                                                   #"\n"))))
          training-outputs (map #(Double/valueOf %) (clojure.string/split (slurp "/home/seditiosus/clojure/linear-regression/ex2y.dat") #"\n"))
          thetas (linear-regression 0.04
                                    training-inputs
                                    training-outputs)
          y1 (hundredths (hypothesis thetas [1 3.5]))
          y2 (hundredths (hypothesis thetas [1 7]))]
      (is (= y1 (float 0.97)))
      (is (= y2 (float 1.2))))))

#_(deftest crickets-test
  (let [data (drop 2 (clojure.string/split (slurp "crickets.csv") #"[\n,]"))
        train-in (map #(Double/valueOf %) (take-nth 2 data))
        train-out (map #(Double/valueOf %) (take-nth 2 (rest data)))]
    (println (univar-linear-regression alpha thetas train-in train-out))))
