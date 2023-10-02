;; # GenLite

;; ## Implementations
;;
;; So far, we have been using what is known as Gen's dynamic modeling language
;; which provides an implementation of the `@gen` macro to define generative
;; functions. In this lecture, we will examine how these work under the hood.
;; First, let's load Gen and the `Distributions` Julia library which provides
;; basic sampling and scoring support for commonly used probability
;; distributions.

(ns lectures.genlite
  (:require [gen.dynamic :as dynamic :refer [gen]]
            [gen.generative-function :as gf]
            [gen.distribution.kixi :as dist]))

;; In order to understand how Gen works, we will introduce a simplified variant
;; of the language. First, instead of using the `@gen` macro to define
;; generative functions, we will define them directly. Amongst other things, the
;; macro is responsible for desugaring the `~` operator. Instead of using this
;; operator, we will assume that we have access to a `sample` function which
;; takes three arguments:
;;
;;    1. the name of the sampled random variable
;;    2. the distribution to sample from and
;;    3. the arguments of the distribution to be sampled from.
;;
;; We will assume that the implementation of the sample function is passed as an
;; argument to our generative function. The `@gen` macro (essentially) desugars
;; the function by adding a similar `sample` function argument. Here are
;; implementations of `flip-biased-coin` in both Gen and GenLite.

(def flip-biased-coin
  (gen [n]
    (let [theta (dynamic/trace! :theta dist/beta 1 1)]
      (mapv (fn [i]
              (dynamic/trace! [:flip i] dist/bernoulli theta))
            (range n)))))

(defn flip-biased-coin-lite [sample n]
  (let [theta (sample :theta dist/beta 1 1)]
    (mapv (fn [i]
            (sample [:flip i] dist/bernoulli theta))
          (range n))))

;; Here is a version of Bayesian linear regression expressed both in terms of Gen and GenLite.

(def line-model
  (gen [xs]
    (let [m   (dist/normal 0 1)
          b   (dist/normal 0 2)
          eps (dist/normal 0 2.5)]
      (doall
       (map-indexed
        (fn [i x]
          (dynamic/trace! [:y i] dist/normal (+ (* m x) b) (* eps eps)))
        xs)))))

(defn line-model-lite [sample xs]
  (let [m   (dist/normal 0 1)
        b   (dist/normal 0 2)
        eps (dist/normal 0 2.5)]
    (doall
     (map-indexed
      (fn [i x]
        (sample [:y i] dist/normal (+ (* m x) b) (* eps eps)))
      xs))))

(into {} (gf/simulate line-model [[1 2 3 4 5]]))

;; In order to make use of these implementations of generative functions, we
;; will need to define `sample` and pass it in. Critical differences in
;; algorithm behavior can be implemented by different version of `sample`.
;;
;; ## Implementing `simulate`
;;
;; Let's see how we can implement a version of `simulate` using this technique.
;;
;; Recall that simulate takes a generative function and its arguments and
;; returns a trace representing a sample from the generative function with those
;; arguments. We will need to represent a few things in this program.
;;
;;  - **The set of random choices (`ChoiceMap` implementation)**. For this, we
;;    will simply use a dictionary (hashtable) with keys being linked
;;    lists (built from pairs) of symbols and/or other datatypes such as `:f =>
;;    1`.
;;
;;  - **The score of the sample**. This will be the sum of the log densities or
;;    probabilities of each random choice made during sampling.
;;
;;  - **The trace**. For this, we will make use of a simple tuple with the following elements:
;;      1. The generative function.
;;      2. The arguments the function was called on.
;;      3. The return value of the function.
;;      4. The set of choices made during sampling.
;;      5. The log probability of the trace that was sampled.
;;
;; ```julia
;; function simulate_lite(gen_func, args)

;;     # initialise the set of choices to an empty dictionary
;;     choices = Dict()

;;     # Initialize the density at 1
;;     score = 0.0

;;     # An implementation of the sample function
;;     function sample_(name, distribution, dist_args)

;;         # Create an instance of the relevant distribution from the Distributions library
;;         dist = distribution(dist_args...)

;;         # Sample the value
;;         value = rand(dist)

;;         # Score the value
;;         density = Distributions.logpdf(dist, value)

;;         # Update the log density with the value
;;         score += density

;;         # Record the sampled value with its name
;;         choices[name] = value

;;         return(value)
;;     end

;;     # Call the generative function with the sample function defined
;;     retval = gen_func(sample_, args...)

;;     # return trace as a named tuple
;;     (gen_func=gen_func,
;;         args=args,
;;         retval=retval,
;;         choices=choices,
;;         score=score)
;; end;

;; simulate_lite(flip_biased_coin_lite, (1000,))


;; simulate_lite(line_model_lite, ([1.,2.,3.,4.,5.],))
;; ```
