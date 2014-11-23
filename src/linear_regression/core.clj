(ns linear-regression.core)

(defn square [x]
  (* x x))

(defn squared-diff [x y]
  (square (- x y)))

(defn univar-hypothesis [theta0 theta1 x]
  (+ (* theta1 x) theta0))

(defn univar-costfn [hypo-y actual-y]
  (let [m (/ 1 (* 2 (count hypo-y)))]
    (* m (reduce + (map squared-diff hypo-y actual-y)))))

;partial derivative of costfn with respect to theta0
(defn univar-dtheta0 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map - hypo-y actual-y)))))

;partial derivative of costfn with respect to theta1
(defn univar-dtheta1 [hypo-y actual-y]
  (let [m (count hypo-y)]
    (* (/ 1 m) (reduce + (map * (map - hypo-y actual-y) hypo-y)))))

(defn univar-batch-gradient-descent [thetas alpha hypo-y actual-y]
  (let [[theta0 theta1] thetas]
    [(- theta0 (* alpha (univar-dtheta0 hypo-y actual-y)))
     (- theta1 (* alpha (univar-dtheta1 hypo-y actual-y)))]))

(defn univar-linear-regression [alpha thetas training-inputs training-outputs]
  (loop [thetas thetas]
    (let [hypo-ys (map (partial univar-hypothesis (first thetas) (last thetas))
                       training-inputs)
          newthetas (univar-batch-gradient-descent thetas alpha hypo-ys
                                            training-outputs)]
      (if (or (Double/isNaN (first thetas))
              (= thetas newthetas))
        thetas
        (recur newthetas)))))

;hypothesis requires the first feature to be 1 always
(defn hypothesis [thetas features]
  (reduce + (map * thetas features)))

(defn costfn [hypo-y actual-y]
  (let [m (count hypo-y)]
    (/ 1 (* 2 m) (reduce + (map squared-diff hypo-y actual-y)))))

;thetas ((t1 t2 t3) (t1 t2 t3))
;features ((f1 f2 f3) (f1 f2 f3))
;hypo-y (1 2)
;actual-y (4 6)
(defn batch-gradient-descent [thetas alpha features hypo-ys actual-ys]
  (let [malpha (* alpha (/ 1 (count hypo-ys)))]
    ;all of this code thinks about thetas wrong
    ;the thetas don't change. One per feature not one per test case
    ;not one group of thetas per group of features
    ;actually this shouldn't be too wrong
    ;maybe it thinks of features wrong I'm tired
    (map (fn [thetas feats]
           (map (fn [tj fj hypo-y actual-y]
                  (let [sumsqdiff (reduce + (map squared-diff hypo-y actual-y))]
                    (- tj (* malpha (* fj sumsqdiff)))))
                thetas feats hypo-ys actual-ys))
         thetas features)))

;training-inputs
;((1 feature feature feature) (1 feature feature feature))
;thetas
;(theta0 theta1 theta2)
(defn linear-regression [alpha training-inputs training-outputs]
  ;i see what I did here, I need to format the data differently since it's still
  ;single var
  (loop [thetas
         (cons 1 (repeatedly (count (first training-inputs)) (constantly 0)))]
    (let [hypo-ys (map (partial hypothesis thetas) (map #(conj % 1) training-inputs))
          new-thetas (batch-gradient-descent thetas alpha
                                            training-inputs hypo-ys training-outputs)]
      ;kay, we got the new-thetas and all that biz, check them for isNaN?
      (if (or (Double/isNaN (last thetas))
              (= thetas new-thetas))
        thetas
        (recur new-thetas)))))
