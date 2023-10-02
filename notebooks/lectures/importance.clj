;; # Statistics Notation
;;
;; In the lectures so far, we have been using a notation for random variables
;; where the RV is capitalized and (regular) variables ranging over values are
;; lower case, so $X=x$. In general, this is uncommon in the statistics
;; literature and you are more likely to see expressions like this:
;;
;; $$p(z\mid x) = \frac{p(x\mid z)p(z)}{p(x)}.$$
;;
;; In this case, we are assuming that $p$ is a density and that $x$ and $z$ are
;; variables with range over values of the random variables $Z$ and $X$, though
;; typically the latter are left implicit and never mentioned explicitly.
;;
;; In general, from now on in the course, we will adopt this convention.
;;
;; Furthermore, we adopt the convention that $z$ ranges over tuples of values
;; for our *latent* random variables, while $x$ range over tuples of values of
;; our *observed* random variables. When we wish to emphasize that our observed
;; are fixed at some particular values (for instance when we are doing
;; inference) we will use a breve like this $\breve{x}$.
;;
;;
;;  - $z$ represents our **latent** random variables.
;;  - $x$ represents our **observed** random variables.
;;  - $\breve{x}$ represents our **observed** random variables fixed at some particular value.
;;  - $p(x,z)=p(x\mid z)p(z)$ represents our **joint distribution** (as a generative model) where.
;;  - $p(x\mid z)$ is the **likelihood** and
;;  - $p(z)$ is the **prior**.
;;  - $p(z\mid x) = \frac{p(x,z)}{p(x)}$ is our **posterior**.
;;  - $p(x) = \int p(x,z) dz$ is the **marginal likelihood** of the observed random variables.
;;
;;

;; ##  Another view of Rejection Sampling: Drawing from a Proposal and "Reweighting"
;;
;; <!-- **TODO: Use lebesgue integrals** = \int p(x) f(x) dx = \lim_{K\to\infty}\frac{1}{K} \sum_{i=1}^{K} f(r_i)-->
;;
;; Sampling is a useful because it allows us to approximately represent complex
;; probability distributions by finite sets of samples from the distribution.
;; For example, if we draw $K$ samples from a random variable $X$ we can use
;; these to approximate arbitrary expectations against $X$, by the law of large
;; numbers.
;;
;; $$\mathbb{E}_{x \sim p} [f(x)]   \approx \frac{1}{K} \sum_{i=1}^{K} f(\hat{x}_i), \quad \hat{x}_i \sim p(x)$$
;;
;;
;; However, as we have seen it is not always possible to efficiently draw exact
;; samples from our target distribution $p$.
;;
;; > What is the central example in this course of a kind of distribution that it is not easy to efficiently draw exact samples from?
;;
;; Rejection sampling is the first example of many algorithm that we will see
;; that takes a different approach: draw samples from an alternative *proposal*
;; distribution $q(x)$ and then reweight the samples to "correct" them to the
;; target distribution $p(x)$ with "correction weight function" $w(x)$.
;;
;; $$\mathbb{E}_{x \sim p} [f(x)]  \approx \frac{1}{\sum_{i=1}^{K} w(\hat{x}_i)} \sum_{i=1}^{K} w(\hat{x}_i) f(\hat{x}_i), \quad \hat{x}_i \sim q(x)$$
;;
;; If we can sample directly from $X \sim p$, then the standard law of large
;; numbers approximation can be thought of as using a weight function which is
;; constant and equal to $1$, that is $w(\cdot)=1$. However, when we draw from
;; another distribution $q(x)$, $w(x)$ needs to be a weighting function
;; that "upweights" samples that are less probable under $q$ than they would be
;; under $p$ and "downweights" samples that more probable under $q$ than they
;; would be under $p$.
;;
;; In rejection sampling the proposal distribution was the joint distribution
;; while the target is the posterior we are trying to approximate. In rejection
;; sampling, we can think of our weights as simply an indicator function that
;; returns whether the condition holds in the sample.
;;
;; $$\mathbb{E}_{p(z\mid \breve{x})} [f(z)] \approx \frac{1}{\sum_{i=1}^{K} \mathbb{1}_{\breve{x}}(\hat{x}_i)} \sum_{i=1}^{K}  \mathbb{1}_{\breve{x}}(\hat{x}_i) f(\hat{z}_i), \quad \hat{x}_i, \hat{z}_i \sim p(x, z)$$
;;
;; In other words, for rejection sampling, our weight function is always either
;; $0$ or $1$ depending on whether the sample matched the conditioner exactly.
;;
;; > Can you think of other possible weight functions?
;;
;; ## Exact Sampling
;;
;; When a sampler returns samples at a rate which is exactly proportional to the
;; target density $p$, we say that it is an *exact* sampler. Rejection sampling,
;; though slow, is an exact sampler? However, in what follows we will see our
;; first cases of another class of sampling algorithms that make use of
;; *approximate samplers*. Let's turn to our first example.

;; # Likelihood Weighting
;;
;; The problem with rejection sampling, we have seen, is that it you may have to
;; wait a long time to draw a sample that satisfies the conditioner. As the
;; marginal likelihood of the conditioner becomes smaller, the time you may have
;; to wait becomes very large. Can we do better than this?
;;
;; One thing to note in our implementation of rejection sampling is that even
;; though we know the condition, we do not make use of this information in any
;; way during generation of the sample. We simply guess and check.
;;
;; One obvious idea is that we might, instead, simply **force** the relevant
;; observations to take on their constrained values and only sample the latent
;; variables in the model? This algorithm is known as *likelihood
;; weighting* (for reasons we will see shortly).
;;
;;
;; And let's look at ourbiased coin model again.
;;
;; ```julia
;; @gen function flip_biased_coin(N)
;;     θ ~ beta(1,1)

;;     [{:x => i} ~ bernoulli(θ)  for i in 1:N]
;; end;
;; ```

;; In Gen, the `generate` function implements likelihood weighting. It is called
;; in a way similar to `simulate` but with an extra argument representing the
;; constraints to be enforced (implemented as a choice map).

;; ```julia
;; observations = Gen.choicemap()

;; observations[:x=>1]=true
;; observations[:x=>2]=true
;; (t, w) = Gen.generate(flip_biased_coin,(3,),observations)
;; Gen.get_choices(t)
;; ```

;; In terms of the graphical model, the likelihood weighting algorithm
;; simply "forces" all shaded nodes to their constrained value and only samples
;; the latent variable nodes.
;;
;; <img src="figures/coin-flipping-bn.jpg"
;;      alt="Biased Coin"
;;      width="100" height="100"/>
;;
;; > What condition must be met for likelihood weighting to work?
;;
;; Note that `generate` also returns an extra return value, a *weight*, which we
;; will talk about shortly.
;;
;; If we substitute `generate` into our rejection sampling code where we
;; formerly used `simulate` what will happen.

;; ```julia
;; function iter_deep(c::Gen.ChoiceMap)
;;   Iterators.flatten([
;;       Gen.get_values_shallow(c),
;;       (Dict(Pair{Any, Any}(k => kk, vv) for (kk, vv) in iter_deep(v))
;;        for (k, v) in Gen.get_submaps_shallow(c))...,
;;   ])
;; end

;; function check_conditions(trace, constraints)
;;     for (name, value) in iter_deep(constraints)
;;         if trace[name] != value return false end
;;     end
;;     return true
;; end

;; function rejection(generative_model, arguments, constraints)
;;     (t,w)=Gen.generate(generative_model,arguments,constraints)
;;     if check_conditions(t,constraints)
;;         print("Matched constraint.")
;;         return(t)
;;     else
;;         print("Failed to match constraint.")
;;         rejection(generative_model, arguments, constraints)
;;     end
;; end

;; observations = Gen.choicemap()

;; for i in 1:10000
;;     observations[:x => i] = true
;; end

;; valid_trace=rejection(flip_biased_coin, (10010,), observations);
;; Gen.get_choices(valid_trace);
;; ```

;; we didn't need to call this in a recursive loop anymore at all, since we match the constraint *every* time.
;;

;; ```julia
;; observations = Gen.choicemap()

;; for i in 1:10000
;;     observations[:x => i] = true
;; end

;; (valid_trace, weight)=Gen.generate(flip_biased_coin, (10010,), observations);
;; Gen.get_choices(valid_trace);
;; ```


;; But now we have a different problem, let's examine the distribution of values of the sampled $\theta$ parameter.

;; ```julia
;; observations = Gen.choicemap()

;; for i in 1:1000
;;     observations[:x => i] = true
;; end

;; histogram([Gen.generate(flip_biased_coin, (1001,),
;;             observations)[1][:θ] for _ in 1:10000])
;; ```

;; Our $\theta$ parameter is still following its prior distribution of a `beta(1,1)`.
;;
;; > Why is this distribution still following the prior?

;; ## Weighted Samples
;;
;; We saw above how simply forcing the observed random variables to take on
;; their constrained values obviously significantly speeds up our sampling
;; algorithm. However, we have a significant problem now. We saw at the end of
;; the last chapter why rejection sampling was a correct sampling algorithm. The
;; algorithm relied on the fact that our target conditional distribution has
;; probability proportional to the joint distribution. By throwing away samples
;; that didn't satisfy the conditioner, we implicitly normalized the
;; distribution on the subset of states in the joint that satisfied the
;; conditioner.
;;
;; However, likelihood weighting no longer satisfies this contract. In
;; particular, when it is possible, the algorithm always returns every sample.
;; Intuitively, many samples from our latent random variables may not be "good"
;; in the sense that they choose values which generate the observed variables
;; with low probability. Moreover, likelihood weighting draws latent random
;; variable values according to the **prior** rather than the posterior.
;;
;; For example, in the biased coin model, likelihood weighting will draw
;; $\theta$ from a $\mathrm{beta}(1,1)$ distribution, even when we observe ten
;; thousand `true` flips as above. In this case, the true posterior over
;; $\theta$ should be steeply peaked near $1$ and, thus, most of our draws from
;; $\mathrm{beta}(1,1)$ will lead to very low probabilities for the observed
;; variables.
;;
;; So why is this useful at all? This is where the weight we mentioned above
;; comes in.
;;
;; In likelihood weighting, we reweight our samples by the likelihood of the
;; observed variables, that is by the probability of the observed variables
;; given the sampled values of the latent variables.
;;
;; Thus, in the example above, if we draw a low $\theta$ this will assign a very
;; low weight to the sample since it generates the condition with low
;; probability.
;;
;; We can represent likelihood weighting mathematically as follows.
;;
;; $$\mathbb{E}_{p(z\mid x)} [f(z)]  \approx  \frac{1}{\sum_{i=1}^{K} p(x \mid \hat{z}_i)} \sum_{i=1}^{K}  p(x \mid \hat{z}_i) f(\hat{z}_i), \quad \hat{z}_i \sim p(z)$$
;;
;; ```julia
;; num_flips=5
;; observations = Gen.choicemap()

;; for i in 1:(num_flips-1)
;;     observations[:x => i] = true
;; end

;; function weighted_sample()
;;     (t,w) = Gen.generate(flip_biased_coin, (num_flips,), observations)
;;     [t[:θ],exp(w)]
;; end

;; samples=[weighted_sample() for _ in 1:10000]

;; thetas = [item[1] for item in samples]
;; weights = [item[2] for item in samples]

;; histogram(thetas, weights=weights)
;; ```

;; # Importance Sampling
;;
;; Likelihood weighting is a special case of a more general weighting method
;; known as *importance sampling*. In importance sampling, we draw from $Y \sim
;; q$ and reweight each sample according to the *normalized importance function*
;; $w(z) = \frac{p(z\mid \breve{x})}{q(z)}$. In other words, we compute our
;; expectations according to the following formula.
;;
;; $$\mathbb{E}_{p(z \mid \breve{x})} [f(z)]  \approx  \frac{1}{\sum_{i=1}^{K} \frac{p(\hat{z}_i \mid \breve{x})}{q(\hat{z}_i)}} \sum_{i=1}^{K}  \frac{p(\hat{z}_i \mid \breve{x})}{q(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim q(z)$$
;;
;; Why does this work?
;;
;; $$\begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)] &= \sum_{z} p(z\mid \breve{x}) f(z)\\
;;                              &= \sum_{z} q(z) \frac{p(z\mid \breve{x})}{q(z)}f(z)\\
;;                              &= \sum_{z} q(z) g(z), \quad g(z) = w(z)f(z) = \frac{p(z\mid \breve{x})}{q(z)}f(z)\\
;;                              &=  \mathbb{E}_{q(z)} [g(z)]\\
;;                              &\approx  \frac{1}{K}\sum_{i=1}^{K} g(\hat{z}_i), \quad \hat{z}_i \sim q(z)\\
;;                              &\approx  \frac{1}{K}\sum_{i=1}^{K}w(\hat{z}_i)f(\hat{z}_i), \quad \hat{z}_i \sim q(z)\\
;;                              &\approx  \frac{1}{K}\sum_{i=1}^{K}\frac{p(\hat{z}_i \mid \breve{x})}{q(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim q(z)\\
;; \end{aligned}$$
;;
;; For importance sampling to work, it is critical that $q(z)>0$ whenever $p(z\mid \breve{x})>0$ (we often say that $q$ is *absolutely continuous* with respect to $p$).
;;
;;
;; The proof above assumed that we can evaluate the density $p(z\mid \breve{x})$
;; exactly. However, in general, for a posterior distribution such as this, we
;; do not know this density exactly (if we did, we could probably sample from it
;; efficiently). What happens when we don't have access to this density, but
;; instead have access to a different density that is proportional to $p(z\mid
;; \breve{x})$ such as the joint $p(\breve{x},z)$ (because $p(z\mid \breve{x}) =
;; \frac{p(\breve{x},z)}{p(\breve{x})}$)?
;;
;; A natural idea would be to use the joint probability $p(\breve{x},z)$ in the
;; numerator of our importance weight. Let's call the new weight function the
;; *unnormalized importance function* $w^*(z)=\frac{p(\breve{x},z)}{q(z)}$.
;;
;;
;; $$\begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)]    &= \sum_{z} p(z\mid \breve{x}) f(z)\\
;;                                   &= \sum_{z} q(z) \frac{p(z\mid \breve{x})}{q(z)}f(z)\\
;;                                   &= \sum_{z} q(z) \frac{p(\breve{x}, z)/p(\breve{x})}{q(z)}f(z)\\
;;                                   &= \sum_{z} q(z) \frac{p(\breve{x}, z)}{p(\breve{x})q(z)}f(z)\\
;;                                    &= \sum_{z} q(z) \frac{1}{p(\breve{x})}\frac{p(\breve{x}, z)}{q(z)}f(z)\\
;;                                   &= \sum_{z} q(z) \frac{1}{p(\breve{x})}w^*(z)f(z)\\
;;                                   &= \sum_{z} q(x) \frac{1}{p(\breve{x})}g(z), \quad g(z) = w^*(z)f(z) = \frac{p(x, z)}{q(z)}f(z) \\
;;                                   &= \mathbb{E}_{q(z)} \left[ \frac{1}{p(\breve{x})} g(z)\right ]\\
;; \mathbb{E}_{p(z\mid x)} [f(z)]    &= \frac{1}{p(\breve{x})}\mathbb{E}_{q(z)} \left[  g(z)\right ]\\
;; p(\breve{x})\mathbb{E}_{p(z\mid x)} [f(z)]&= \mathbb{E}_{q(z)} \left[  g(z)\right ]\\
;;                                   &\approx \frac{1}{K}\sum_{i=1}^{K} g(\hat{z}_i), \quad \hat{z}_i \sim q\\
;;                                   &\approx \frac{1}{K}\sum_{i=1}^{K} w^*(z)f(\hat{z}_i), \quad \hat{z}_i \sim q\\
;;                                   &\approx \frac{1}{K}\sum_{i=1}^{K}\frac{p(\breve{x}, z)}{q(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim q\\
;; \end{aligned}$$
;;
;; So now we have a problem. By using an unnormalized version of our target
;; distribution $p$ in our modified unnormalized importance weight function
;; $w^*(\cdot)$ our importance sampling approximator does not give us the
;; correct expectation, but rather an expectation which is off by a factor of
;; $p(x)$&mdash;the marginal likelihood. In order to correct this problem, we
;; need to divide both sides by the marginal likelihood. But, of course, we
;; don't know the marginal likelihood, if we did, we wouldn't need to use
;; importance sampling in the first place!!
;;
;; So, how can we correct this issue? There is an important fact about our
;; importance weights which can come to the rescue. Let's see how their
;; expectation behaves with respect to the proposal distribution $q$.
;;
;; However, let's look at the expectation of our importance weight with
;; unnormalized target density such as the joint $p(x, z)$.
;;
;; $$ \begin{aligned}
;; \mathbb{E}_{q(z)}\left [w^*(z)\right] &= \mathbb{E}_{q(z)}\left [\frac{p(\breve{x}, z)}{q(z)} \right]\\
;; &=\mathbb{E}_{q(z)}\left [\frac{p(\breve{x}, z)}{q(z)} \right]\\
;; &=\sum_{z} q(z) \frac{p(\breve{x}, z)}{q(z)}\\
;; &=\sum_{z} p(\breve{x}, z)\\
;; &= p(\breve{x})\\
;; \end{aligned}
;; $$
;;
;; Thus, we can estimate our marginal likelihood $p(x)$ by averaging our
;; importance weights.
;;
;; $$ \begin{aligned}
;; p(x) &= \mathbb{E}_{q(z)}\left [w^*(z)\right]\\
;; &\approx \frac{1}{K} \sum_{i=1}^{K}  w^*(\hat{z}_i), \quad \hat{z}_i \sim q\\
;; &\approx \frac{1}{K} \sum_{i=1}^{K}  \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}, \quad \hat{z}_i \sim q\\
;; \end{aligned}
;; $$
;;
;;  This suggests that we can use monte carlo to correct the problem above.
;;
;; $$\begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)]  &= \sum_{z} p(z\mid \breve{x}) f(z)\\
;;                                 &= \sum_{z} q(z) \frac{p(z\mid x)}{q(z)}f(z)\\
;;                                 &= \sum_{z} q(z) \frac{p(\breve{x},z)/p(x)}{q(z)}f(z)\\
;;                                 &= \sum_{z} q(z) \frac{1}{p(\breve{x})}g(z), \quad g(z) = w^*(z)f(z) = \frac{p(\breve{x}, z)}{q(z)}f(z) \\
;;                                 &= \mathbb{E}_{q(z)} \left[ \frac{1}{p(\breve{x})} g(z)\right ] \\
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)]  &= \frac{1}{p(\breve{x})}\mathbb{E}_{q(z)} \left[  g(z)\right ]\\
;;                                 &= \frac{1}{\mathbb{E}_{q(z)}[w^*(z)]}\mathbb{E}_{q(z)} \left[  g(z)\right ]\\
;;                                 &\approx  \frac{1}{\frac{1}{K}\sum_{i=1}^{K} w^*(\hat{z}_i)}  \frac{1}{K}\sum_{i=1}^{K}g(\hat{z}_i), \quad \hat{z}_i \sim q\\
;;                                 &\approx  \frac{1}{\sum_{i=1}^{K} w^*(\hat{z}_i)}\sum_{i=1}^{K}g(\hat{z}_i), \quad \hat{z}_i \sim q\\
;;                                 &\approx  \frac{1}{\sum_{i=1}^{K} \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}  \sum_{i=1}^{K}\frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim q\\
;;                                 &\approx  \sum_{i=1}^{K}\frac{\frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}{\sum_{i=1}^{K} \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}f(\hat{z}_i), \quad \hat{z}_i \sim q\\
;; \end{aligned}$$
;;
;; $\frac{\frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}{\sum_{i=1}^{K} \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}$ is called the *self-normalized importance weight*.  Note that, of course and critically, we must be able to sample from $\hat{z}_i \sim q$. Also note that we are now estimating our original expectation as the ratio of two estimators. The ratio of two unbiased estimators is biased and, in practice, will tend to underestimate the true quantity.
;;
;; Let's look at some examples of importance sampling.

;; ## Exact Sampling from the Posterior
;;
;; What if we can take exact samples from our posterior distribution?
;;
;; In this case $q$ is equal to our exact posterior $p$. Using the normalized
;; form of importanc sampling, we get
;;
;;
;; <!-- $$ \begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)]
;; &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\hat{z}_i\mid \breve{x})}{p(\hat{z}_i\mid \breve{x})}} \sum_{i=1}^{K}\frac{p(\hat{z}_i\mid \breve{x})}{p(\hat{z}_i\mid \breve{x})} f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid x)\\
;; &\approx \frac{1}{\sum_{i=1}^{K} 1} \sum_{i=1}^{K} 1 f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; &\approx \frac{1}{K} \sum_{i=1}^{K} f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; \end{aligned}.
;; $$ -->
;;
;; $$ \begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)]
;; &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\hat{z}_i \mid \breve{x})p(\breve{x})}{p(\hat{z}_i\mid \breve{x})}} \sum_{i=1}^{K}\frac{p(\hat{z}_i\mid \breve{x})p(\breve{x})}{p(\hat{z}_i\mid \breve{x})} f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; &\approx \frac{1}{\sum_{i=1}^{K} p(\breve{x})} \sum_{i=1}^{K} p(\breve{x})f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; &\approx \frac{p(\breve{x})}{p(\breve{x})\sum_{i=1}^{K} 1} \sum_{i=1}^{K} 1 f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; &\approx \frac{1}{\sum_{i=1}^{K} 1} \sum_{i=1}^{K} 1 f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; &\approx \frac{1}{K} \sum_{i=1}^{K} f(\hat{z}_i), \quad \hat{z}_i \sim p(z\mid \breve{x})\\
;; \end{aligned}.
;; $$
;;
;; So in this case, our importance sampling principle just reduces to the
;; standard Monte Carlo approximation.

;; ## Rejection Sampling
;;
;; In this case, our target posterior $p$ cannot be evaluated. However, we can
;; evaluate the joint distribution, which we can write as the product of the
;; likelihood and prior. We take samples from this joint, so our proposal here
;; is over both our latent variables $z$ **and** our observed variables $x$. We
;; can write this proposal like this
;;
;; $$q(x,z)=p(x\mid z)p(z).$$
;;
;; The score of each sample under the true posterior is proportional to $p(x\mid
;; z)p(z)$ for samples which satisfy the conditioner $\breve{x}$ but it is $0$
;; otherwise. Thus, each sample $x$ has posterior probability proportional to
;; our target distribution:
;;
;; $$p(x\mid z)p(z)\mathbb{1}_{\breve{x}}(x)$$
;;
;; So,
;; $$ \begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)] &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\hat{x}_i\mid \hat{z}_i)p(\hat{z}_i)\mathbb{1}_{\breve{x}}(\hat{x}_i)}{p(\hat{x}_i\mid \hat{z}_i)p(\hat{z}_i)}} \sum_{i=1}^{K}  \frac{p(\hat{x}_i\mid \hat{z}_i)p(\hat{z}_i)\mathbb{1}_{\breve{x}}(\hat{x}_i)}{p(\hat{x}_i\mid \hat{z}_i)p(\hat{z}_i)} f(\hat{z}_i), \quad \hat{x}_i, \hat{z}_i \sim p(x\mid z)p(z)\\
;; &\approx \frac{1}{\sum_{i=1}^{K} \mathbb{1}_{\breve{x}}(\hat{x}_i)} \sum_{i=1}^{K}  \mathbb{1}_{\breve{x}}(\hat{x}_i) f(\hat{z}_i), \quad \hat{x}_i, \hat{z}_i \sim p(x\mid z)p(z)\\
;; \end{aligned}
;; $$

;; ## Likelihood Weighting
;;
;; In this case, as in rejection sampling, our target posterior $p$ cannot be
;; evaluated, but we can evaluate the joint distribution. We force our samples
;; to take on the desired values in the likelihood, so our our proposal is just
;; the prior distribution:
;;
;; $$q(z)=p(z)$$
;;
;; The score of each sample under the true posterior is proportional to $p(x\mid
;; z)p(z)$ for all samples (since we are forcing the conditioner to be
;; satisfied) so our target distribution is:
;;
;; $$p(\breve{x}\mid z)p(z)$$
;;
;; So,
;; $$ \begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)] &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\breve{x}\mid \hat{z}_i)p(\hat{z}_i)}{p(\hat{z}_i)}} \sum_{i=1}^{K}  \frac{p(\breve{x}\mid \hat{z}_i)p(\hat{z}_i)}{p(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim p(z)\\
;; &\approx \frac{1}{\sum_{i=1}^{K} p(\breve{x} \mid \hat{z}_i)} \sum_{i=1}^{K}  p(\breve{x} \mid \hat{z}_i) f(\hat{z}_i), \quad \hat{z}_i \sim p(z)\\
;; \end{aligned}
;; $$

;; # Importance Resampling
;;
;; *Importance sampling* isn't really an inference algorithm, rather it is more
;; a framework for thinking about how to mathematically relate samples from one
;; distribution to another distribution by taking samples from a proposal
;; distribution $q$ and assigning them weights $p/q$. How can we use importance
;; sampling to generate samples (approximately) from our target distribution?
;;
;; One idea is known as *importance resampling*. Our importance weights tell us
;; how to upweight or downweight our samples to make them match our target
;; distribution. We can approximate the true distribution my renormalizing these
;; weights and sampling a trace from them.

;; ```julia
;; function my_importance_resampling(model,arguments,constraints,computation)
;;     samples=[Gen.generate(model,arguments,constraints) for i=1:computation]
;;     traces = [item[1] for item in samples]
;;     weights = [item[2] for item in samples]

;;     weight_sum=logsumexp(weights)
;;     normed_ws=weights.-weight_sum

;;     average_weight=weight_sum-log(length(weights))
;;     result=traces[categorical(exp.(normed_ws))]
;;     (result, average_weight)
;; end

;; observations = Gen.choicemap()

;; num=100
;; for i in 1:num
;;     observations[:x => i] = true
;; end

;; ss=[my_importance_resampling(flip_biased_coin,
;;                              (num+1,),
;;                              observations,
;;                              500)
;;     for _ in 1:1000]

;; θs = [x[1][:θ] for x in ss]
;; histogram(θs)
;; ```

;; One way to view importance resampling is as an estimator for posterior
;; probabilities by considering the function that we are taking expecatations
;; of (usually $f$) to be the indicator function at each sampled datapoint.
;;
;; $$\begin{aligned}
;; \mathbb{E}_{p(z\mid \breve{x})} [f(z)] &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}  \sum_{i=1}^{K}\frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)} f(\hat{z}_i), \quad \hat{z}_i \sim q \\
;; \mathbb{E}_{p(z\mid \breve{x})} [\mathbb{1}_{z^\prime}(z)] = p(z^\prime\mid \breve{x})  &\approx \frac{1}{\sum_{i=1}^{K} \frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)}}  \sum_{i=1}^{K}\frac{p(\breve{x}, \hat{z}_i)}{q(\hat{z}_i)} \mathbb{1}_{z^\prime}(\hat{z}_i), \quad \hat{z}_i \sim q\\
;; \end{aligned}$$
