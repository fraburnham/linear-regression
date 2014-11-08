(ns linear-regression.core)

(defn square [x]
  (* x x))

(defn squared-diff [x y]
  (square (- x y)))

;so the hypofn can be used for map
;but i have the others taking in a list, not very "programming to abstraction"
;so what can I do to improve it? dtheta0 is basically a wrapper
;from the "math" view it looks right the costfn includes the summation
;the derivatives include the summation...

;the hypothesis for our univariate linear regression
;h(x) = theta0 + theta1(x)
(defn hypothesis [theta0 theta1 x]
  (+ (* theta1 x) theta0))

;cost function we're using is the sum of squared diffs
(defn costfn [hypo-y actual-y]
  (let [m (/ 1 (* 2 (count hypo-y)))]
    (* m (reduce + (map squared-diff hypo-y actual-y)))))

;partial derivative of costfn with respect to theta0
;1/m(sum(hypo(xi)-yi))
(defn dtheta0 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map - hypo-y actual-y)))))

;partial derivative of costfn with respect to theta1
;1/m(sum((hypo(xi) - yi) * xi))
(defn dtheta1 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map * (map - hypo-y actual-y) hypo-y)))))

;gradient descent
;update the given theta
;theta-alpha*derWithRespectToTheta
;returns the two new thetas that we'll use
(defn batch-gradient-descent [thetas alpha hypo-y actual-y]
  (let [theta0 (first thetas)
        theta1 (last thetas)]
    [(- theta0 (* alpha (dtheta0 hypo-y actual-y)))
     (- theta1 (* alpha (dtheta1 hypo-y actual-y)))]))

;so the actual loop logic will be to
;start with thetas at zero
;map the hypo over them and our "x" list/vector/whatever ;) go map ;)
;go straight to gradient descent
;if the thetas are unchanged then you've got your hypothesis!
(defn univar-linear-regression [alpha thetas training-inputs training-outputs]
  (loop [thetas thetas]
    (let [hypo-ys (map (partial hypothesis (first thetas) (last thetas))
                       training-inputs)
          newthetas (batch-gradient-descent thetas alpha hypo-ys 
                                            training-outputs)]
      (if (or (Double/isNaN (first thetas))
              (and (= (first thetas) (first newthetas))
                   (= (last thetas) (last newthetas)))) thetas
               (recur newthetas)))))

;there is some data on how to pick a good alpha
;http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex3/ex3.html
;read up and improve

;this pulls the housing data out from andrew ng's course

(defn test [alpha thetas] 
  (let [training-inputs (map #(Double/valueOf %) (clojure.string/split (slurp "ex2x.dat") #"\n"))
      training-outputs (map #(Double/valueOf %) (clojure.string/split (slurp "ex2y.dat") #"\n"))]
  (println (univar-linear-regression alpha thetas 
                                     training-inputs training-outputs))))
