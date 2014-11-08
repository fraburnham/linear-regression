# Introduction to linear-regression

#(square [x])

Returns (* x x).

#(squared-diff [x y])

Returns (square (- x y)). This is used in the cost function as the "squared 
error".

#(hypothesis [theta0 theta1 x])

Returns (+ theta0 (* theta1 x)). Theta0 and theta1 are the parameters 
hypothesis uses to plot the line. Once we've minimized the error in the thetas
the hypothesis function is used to predict new values.

#(costfn [hypo-y actual-y])

Returns the cost found by summing the squared error and reducing by
(/ 1 (* (count hypo-y) 2)). Expects hypo-y and actual-y to be sequences.

#(dtheta0 [hypo-y actual-y])

Returns the derivative of hypothesis with respect to theta0.
Expects hypo-y and actual-y to be sequences.

#(dtheta1 [hypo-y actual-y])

Returns the derivative of hypothesis with respect to theta1.
Expects hypo-y and actual-y to be sequences.

#(batch-gradient-descent [thetas alpha hypo-y actual-y])

Returns new values for thetas after performing gradient descent.
Expects hypo-y and actual-y to be sequences.

#(univar-linear-regression [alpha thetas training-inputs training-outputs])

Returns values for theta0 and theta1 that can be used with hypothesis.
Will return NaN on divergence. There are cases where it will "dance" around
the solution instead of converging.
