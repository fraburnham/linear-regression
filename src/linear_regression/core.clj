(ns linear-regression.core)

(defn square [x]
  (* x x))

(defn squared-diff [x y]
  (square (- x y)))

;the hypothesis for our univariate linear regression
;h(x) = theta0 + theta1(x)
(defn hypothesis [theta0 theta1 x]
  (+ (* theta1 x) theta0))

(defn costfn [hypo-y actual-y]
  (let [m (/ 1 (* 2 (count hypo-y)))]
    (* m (reduce + (map squared-diff hypo-y actual-y)))))

;partial derivative of costfn with respect to theta0
(defn dtheta0 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map - hypo-y actual-y)))))

;partial derivative of costfn with respect to theta1
(defn dtheta1 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map * (map - hypo-y actual-y) hypo-y)))))

(defn batch-gradient-descent [thetas alpha hypo-y actual-y]
  (let [theta0 (first thetas)
        theta1 (last thetas)]
    [(- theta0 (* alpha (dtheta0 hypo-y actual-y)))
     (- theta1 (* alpha (dtheta1 hypo-y actual-y)))]))

(defn univar-linear-regression [alpha thetas training-inputs training-outputs]
  (loop [thetas thetas]
    (let [hypo-ys (map (partial hypothesis (first thetas) (last thetas))
                       training-inputs)
          newthetas (batch-gradient-descent thetas alpha hypo-ys 
                                            training-outputs)]
      (println (costfn hypo-ys training-outputs))
      (if (or (Double/isNaN (first thetas))
              (and (= (first thetas) (first newthetas))
                   (= (last thetas) (last newthetas)))) thetas
               (recur newthetas)))))

;there is some data on how to pick a good alpha
;http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
;read up and improve

;without reading up on anything it seems to me that I can start with like 10
;it'll diverge to NaN farily quickly, or converge on a solution
;divide by half or something and re-run until you get convergence
;it may get stuck in back and forth loops. Dancing around the answer
;so perhaps a scaling alpha isn't a bad idea? Either way there needs to
;be some checking for conditions where it'll never converge and we're wasting
;cpu time.

;write some proper tests for this library.
;it works in the two cases tested

(defn height-test [alpha thetas] 
  (let [training-inputs (pmap #(Double/valueOf %) (clojure.string/split (slurp "
ex2x.dat") #"\n"))
      training-outputs (pmap #(Double/valueOf %) (clojure.string/split (slurp "ex2y.dat") #"\n"))]
  (println (univar-linear-regression alpha thetas 
                                     training-inputs training-outputs))))


(defn test [alpha thetas]
  (let [data (drop 2 (clojure.string/split (slurp "crickets.csv") #"[\n,]"))
        train-in (map #(Double/valueOf %) (take-nth 2 data))
        train-out (map #(Double/valueOf %) (take-nth 2 (rest data)))]
    (println (univar-linear-regression alpha thetas train-in train-out))))
