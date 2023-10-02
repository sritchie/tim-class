;; In this course, our fundamental inference problem is to find ways of
;; representing posterior distributions&mdash;typically, we have access to a
;; joint distribution over some combination of latent and observed random
;; variables and wish to compute the posterior distribution over the latents
;; given the observed variables. In general, this is intractable as it involves
;; computing the marginal likelihood of the latent variables given the observed
;; variables, which typically requires evaluating intractable sums and
;; integrals.
;;
;; So far, we have seen one family of approximate inference techniques for this
;; problem using various kinds of *importance sampling*. The idea behind
;; importance sampling is that although we cannot sample from our target
;; distribution $p$ we have access to a proposal distribution over the same
;; space, $q$, such that $q(\mathbf{v})>0$ whenever $p(\mathbf{v}) > 0$. By
;; weighting our samples from $q$ we can correct these samples to the true
;; distribution $p$.
;;
;; Importance sampling has some serious limitations, however, which will
;; necessistate the introduction of some more advanced techniques. In this unit,
;; we will look at a family of techniques known as *Markov chain Monte Carlo*.
;; But first, let's motivate this with an example.

;; ```julia
;; using Gen
;; using Distributions
;; using GenDistributions
;; using Plots
;; using Clustering
;; ```

;; ```julia
;; const dirichlet = DistributionsBacked(
;;     alpha -> Dirichlet(alpha),
;;     (true,),
;;     true,
;;     Vector{Float64})

;; const trunc_normal = DistributionsBacked(
;;     (mu, sigma) -> TruncatedNormal(mu, sigma, 0, Inf),
;;     (true,true),
;;     true,
;;     Float64)
;; ```

;; # Mixture Models
;;
;; One fundamental problems that frequently arises in statistics, artificial
;; intelligence, machine learning, and computational cognitive science is that
;; of putting observations into categories. For instance, in speech perception
;; we must learn to recognize different segments of acoustic data into
;; categories of speech sounds, such as /p/ or /a/. If we do not know the set of
;; possible categories in advance&mdash;as children do not&mdash;this is known
;; as the problem of *clustering*.
;;
;; In Bayesian statistics, clustering is handled by a class of models known as
;; *mixture moodels*. Mixture models all consist of a basic generative template.
;; We start by assuming we know the number of target categories (clusters)
;; $K$&mdash;known as *components* in generative modeling. However, we do not
;; know how datapoints are assigned to components in advance, how likely each
;; component is, nor the properties of the components, such as how each category
;; predicts the acoustics of the sound it represents.
;;
;; To capture the properties of each category, will assume that each component is
;; associated with an *observation* distribution; that is, a distribution over
;; the type of data we are generating for that cluster. For example, in the case
;; of speech data, which is typically represented using (sequences of) vectors of
;; real numbers, our observation distribution could  be some distribution over
;; (sequences of) vectors drawn from some density on $\mathbb{R}^n$, such as a
;; multivariate Gaussian (see below).
;;
;; In a mixture model, we use the following schematic steps to sample each
;; datapoint.
;;
;; 1. Draw a category from a *mixing distribution* over our $K$ categories.
;;
;; 2. Look up the observation distribution for that category.
;;
;; 3. Sample an observation from the *observation/emission distribution*
;;    particular category.
;;
;; To make this concrete, of course. We will need to find specific distributions
;; implementations for our mixing distribution and observation distributions, and
;; potentially, any prior distributions we might put on parameters of these
;; models.
;;
;; Below is the code for our Gaussian mixture model that we saw in the first
;; problem set. Let's use it to generate an artificial dataset which we can
;; condition on, and explore the behavior of the importance sampler in this
;; setting.

;; ```julia
;; @gen function gmm(K,T)
;;     # set hyperparameters
;;     δ = 5 * ones(K)
;;     ξ = 0.0
;;     κ = 0.01
;;     α = 2.0
;;     β = 10.0

;;     # Draw the mixture weights
;;     ϕ ~ dirichlet(δ)

;;     # Draw the means and variances for each component distribution
;;     means, vars = Vector{Real}(zeros(K)), Vector{Real}(zeros(K))
;;     for j=1:K
;;         means[j] = ({:μ => j} ~ normal(ξ, 1/κ))
;;         vars[j] = ({:σ² => j} ~ inv_gamma(α, β))
;;     end

;;     # Draw the observations
;;     xs = zeros(T)
;;     zs = Array{Int64}(undef,T)
;;     for i=1:T
;;         zs[i] = ({:z => i} ~ categorical(ϕ))
;;         xs[i] = {:x => i} ~ normal(means[zs[i]], vars[zs[i]])
;;     end
;;     (xs, zs)
;; end
;; ```

;; Let's sample 5000 datapoints from a mixture model with 10 components and
;; render it.

;; ```julia
;; gaussian_pdf(μ, σ²) = x -> exp(Gen.logpdf(normal, x, μ, σ²));
;; function render_gmm_trace(tr)
;;     K, T = get_args(tr)

;;     plot()
;;     for k=1:K
;;         μ, σ² = tr[:μ => k], tr[:σ² => k]
;;         # plot the gaussian density for component k (mean ±7SD)
;;         plot!((μ - 7sqrt(σ²)):1e-1:μ + 7sqrt(σ²), gaussian_pdf(μ, σ²),
;;             color=k, ls=:dash)
;;         x_ks = [tr[:x => t] for t=1:T if tr[:z => t] == k]
;;         if length(x_ks) > 0
;;             # plot a histogram of observations in this component
;;             histogram!(x_ks, normalize=:pdf, alpha=0.25, lα=0, color=k)
;;             # plot x points as a jittered rug
;;             scatter!(x_ks, _->-1e-3(k+1),
;;                 color= k, ms=2, mα=.5, shape=:vline)
;;         end
;;     end
;;     plot!(xlabel="x", ylabel="density", legend=false,
;;         title="Gaussian mixture model")
;; end;
;; true_trace=Gen.simulate(gmm, (10, 5000))
;; render_gmm_trace(true_trace)
;; ```

;; ```julia jupyter={"outputs_hidden": true} tags=[]
;; get_choices(true_trace)[:z => 306]
;; ```


;; Let's draw an importance sample from the conditional distribution on cluster
;; assignments using our artificial dataset.

;; ```julia
;; function get_constrained_trace(true_trace)
;;     K, T = get_args(true_trace)

;;     observations = Gen.choicemap()

;;     observations[:ϕ] = true_trace[:ϕ]

;;     for j=1:K
;;         observations[:μ => j] =  true_trace[:μ => j]
;;         observations[:σ² => j] =  true_trace[:σ² => j]
;;     end

;;     for i in 1:T
;;         observations[:x => i] = true_trace[:x => i]
;;     end

;;     (t,w) = Gen.generate(gmm, (K,T), observations)

;; end

;; constrained_trace,w=get_constrained_trace(true_trace)
;; render_gmm_trace(constrained_trace)
;; w
;; ```

;; ```julia
;; get_choices(constrained_trace)[:z => 306]
;; ```

;; Of course, the actual value for the cluster assignment is not meaningful
;; since renaming each cluster with a different integer is the same solution. We
;; need a measure of how often observations from the same cluster in our
;; predicted data end up in the same cluster in our true data and vice versa.
;; One convenient information-theoretic metric for this is known as the
;; *v-measure*. It varies between 0 (clusterings are maximally different) and
;; 1 (two clusterings are identical).

;; ```julia
;; function compute_v_measure(t1, t2)
;;     zs1=Gen.get_retval(t1)[2]
;;     zs2=Gen.get_retval(t2)[2]
;;     Clustering.vmeasure(zs1,zs2)
;; end

;; num_samples=2000
;; ms = zeros(num_samples)
;; ws = zeros(num_samples)
;; wms = zeros(num_samples)


;; for i in 1:num_samples
;;     t, ws[i] = get_constrained_trace(true_trace)
;;     ms[i]=compute_v_measure(t,true_trace)
;; end

;; norm=maximum
;; nws=ws.-norm(ws)
;; nwms = exp.(log.(ms .* exp.(nws)).+ norm(ws))

;; (compute_v_measure(true_trace,true_trace),
;;     mean(ms),
;;     mean(nwms))
;; ```

;; Even the importance weighted mean of the v-measure score is hopeless. The
;; problem is that since we are drawing a large number of latent cluster
;; assignments from the proposal distribution, it is very unlikely that such
;; assignments are correct for more than a handful of cases. How can we fix
;; this?
;;
;; Markov Chain Monte Carlo is based on the idea that instead of updating the
;; entire set of latent random variables in one go, we can instead update a few
;; at a time. This way, we can get "partial credit" for parts of the latent
;; space where are answers are already good. Let's consider how to do this for
;; our cluster assigments above.

;; # Gibbs Sampling
;;
;; The basic idea of Gibbs sampling is that we will update one latent variable
;; at a time, while leaving the rest fixed. Let's see how we might do this in
;; the case of our mixture model.
;;
;; The basic Gibbs sampling algorithm works as follows.
;;
;; 1. Create an intial trace of the target model.
;; 2. Choose a latent random variable $z_i$ in that trace that you would like to resample.
;; 3. Enumerate all of the possibile values for $z_i$ and for each:
;;     3a. Construct the trace that results from substituting that value of $z_i$ into the original trace.
;;     3b. Score the new trace with the substituted value of $z_i$.
;; 4. Renormalize the scores from the resulting set of traces.
;; 5. Sample one of the traces and set that as your new initial trace and return to 1.
;;

;; ```julia
;; function gibbs_dist(trace, addr, values)
;;     lps = Vector{Float64}(undef, length(values))
;;     for (i, val) in enumerate(values)
;;         (t, _) = Gen.update(trace, choicemap((addr, val)))
;;         lps[i]=get_score(t)
;;     end
;;     exp.(lps .- logsumexp(lps))
;; end

;; function gibbs_update(trace, addr, values)
;;     probs = gibbs_dist(trace, addr, values)
;;     idx = categorical(probs)
;;     val = values[idx]
;;     new_trace, _ = Gen.update(trace, choicemap((addr, val)))
;;     return new_trace
;; end

;; K, T = get_args(true_trace)

;; new_trace=gibbs_update(constrained_trace, :z => 306,1:K)
;; (new_trace[:μ => 1], constrained_trace[:μ => 1], new_trace[:z => 306], constrained_trace[:z => 306])
;; ```

;; These functions have made use of a new member of the generative function
;; interface: `Gen.update`. Update, takes a trace and a choicemap of constrained
;; values and updates the original trace with the new values&mdash;doing all
;; necessary bookeeping to update the scores. `update` will also sample from the
;; prior any random variables which were not specified in either the original
;; trace of the constraint choicemap.
;;
;; > What circumstances could cause it to be necessary to sample value for
;; > random variables not in the original trace or constraint choicemap?
;;
;; Often we use a Gibbs sampler in *sweeps* over all of the target latent random
;; variables we would like to resample.

;; ```julia
;; function gmm_gibbs_sweep(trace, addr_prefix)

;;     K, T = get_args(trace)

;;     # Update each latent variable in turn
;;     for i=1:T
;;         trace = gibbs_update(trace, addr_prefix=>i, 1:K)
;;     end

;;     return trace
;; end
;; ```

;; Let's see what happens when we run a single Gibbs sweep over all of our
;; component assigment variables $z_i$ in our GMM model.

;; ```julia
;; initial_trace,w=get_constrained_trace(true_trace)
;; render_gmm_trace(initial_trace)

;; sweep_trace=gmm_gibbs_sweep(initial_trace, :z)
;; render_gmm_trace(sweep_trace)
;; ```

;; We can also look at the improvement in the v-measure of our discovered
;; solution after one sweep.

;; ```julia
;; (compute_v_measure(true_trace,true_trace),
;;     compute_v_measure(true_trace,initial_trace),
;;    compute_v_measure(true_trace,sweep_trace))
;; ```

;; The improvement is clear. Typically, Gibbs samplers work by running some
;; number of sweeps over the entire model.

;; ```julia
;; function gmm_gibbs_sampler(true_trace,sweeps)
;;    trace,w = get_constrained_trace(true_trace)

;;    for i=1:sweeps
;;         trace = gmm_gibbs_sweep(trace, :z)
;;    end

;;    return trace
;; end

;; final_trace=gmm_gibbs_sampler(true_trace,10)
;; render_mixture_trace(final_trace)
;; ```

;; ```julia
;; (compute_v_measure(true_trace,true_trace),
;;    compute_v_measure(true_trace,final_trace))
;; ```

;; # The Gibbs Distribution
;;
;; We have defined our Gibbs sampler heuristically by substituting in the values
;; for some random variable, getting the score of the resulting trace, and
;; sampling from this set of scores renormalized. But what distribution is this?
;; Let's remind ourselves of the joint distribution for the GMM model.
;;
;; $$\begin{align}
;; p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:T},x_{1:T} \mid
;;     \delta, \xi, \kappa, \alpha, \beta)
;; =
;; &p(\phi \mid \delta)
;; \left [ \prod_{k=1}^K
;;     p(\mu_k \mid \xi,\kappa )
;;     p(\sigma^2_k \mid \alpha, \beta )
;; \right ]
;; \left[ \prod_{t=1}^T
;;     p(z_t | \phi)
;;     p(x_t \mid \mu_{z_t}, \sigma^2_{z_t})
;; \right]
;; \end{align}$$
;;
;; Recall that the score of a particular trace is just this expression with each
;; of its variables instantiated by some particular value. Let's assume we are
;; considering the component assignment of data point $306$, that is, $z_{306}$.
;; It is useful to rewrite our joint probability pulling out the terms for this
;; datapoint.
;;
;;
;; $$\begin{align}
;; &p(z_{306} | \phi)p(x_{306} \mid \mu_{z_{306}}, \sigma^2_{z_{306}}, z_{306}) \times \\
;; &p(\phi \mid \delta)
;; \left [ \prod_{k=1}^K
;;     p(\mu_k \mid \xi,\kappa )
;;     p(\sigma^2_k \mid \alpha, \beta )
;; \right ]
;; \left[ \prod_{t=1}^{305}
;;     p(z_t | \phi)
;;     p(x_t \mid \mu_{z_t}, \sigma^2_{z_t})
;; \right]
;; \left[ \prod_{t=307}^{T}
;;     p(z_t | \phi)
;;     p(x_t \mid \mu_{z_t}, \sigma^2_{z_t})
;; \right]
;; \end{align}$$
;;
;; Note here that we are using the convention that any conditional or marginal
;; densities written with $p$ are assumed to be those related to the joint
;; written with $p$ via marginalization and/or conditioning.
;;
;; For clarity, we can pull out the term corresonding the the distribution on
;; $z_{306}$
;;
;; $$p(z_{306} | \phi, x_{306}, \mu_{306}, \sigma^2_{306})p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})$$
;;
;; Note that for some particular value of $z_{306}$, say $z_{306}=2$ (component
;; $2$), this represents the score of our trace with that value substituted in.
;; What if we renormalize this probability over all possible values of
;; $z_{306}$?
;;
;; $$\begin{align}
;; =&\frac{p(z_{306} | \phi, x_{306}, \mu_{306}, \sigma^2_{306}) p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}
;; {\sum_{z_{306}} p(z_{306} | \phi, x_{306}) p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}\\
;; =&\frac{p(z_{306}, \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}
;; {\sum_{z_{306}} p(z_{306}, \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}
;; \\
;; =&\frac{p(z_{306}, \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}
;; {p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})}\\
;; =&p(z_{306} | \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:305},z_{307:T},x_{1:T})
;; \end{align}$$
;;
;; In other words, by renormalizing the probability of each trace in gibbs
;; sampling, we are in fact sampling from the conditional distribution on
;; $z_{306}$ given **all other observed and latent variables**. This conditional
;; distribution is sometimes called the *Gibbs distribution*, the *full
;; conditional* or the *conditional posterior* distribution. This last term is
;; sometimes used because this is a posterior since it conditions on the
;; observed variables, but it also happens to condition on particular values for
;; some of the latent variables as well.

;; # Approximating the True Posterior via Gibbs Sampling
;;
;; We have seen that our heuristic of renormalizing traces actually is the same
;; thing as sampling from the conditional posterior over the target variable.
;; The Gibbs sampler loops over the set of target latent variables, computing
;; the conditional posterior of each in turn, and sampling from it and updating
;; the trace. This seems like an intuitively reasonable thing to do, but is it
;; actually correct? We would like to sample from the *joint posterior* over our
;; component assignments.
;;
;; $$p(z_{1:T} | \phi, \mu_{1:K}, \sigma^2_{1:K}, x_{1:T})$$
;;
;; But instead we are sampling in turn from each *conditional posterior*.
;;
;; $$p(z_{i} | \phi, \mu_{1:K}, \sigma^2_{1:K},z_{1:(i-1)},z_{(i+1):T}, x_{1:T})$$
;;
;; Is there some sense in which this process is a correct approximation to the
;; desired *joint posterior* over all of the target variables at once? It turns
;; out that the answer is yes!
;;
;; But to understand why, we need to introduce a new concept: *Markov chains*.

;; # Markov Chains
;;
;; Let $\mathcal{\mathbf{v}}$ be a general state space, that is some set of
;; objects&mdash;for instance, it could be the set of integers, the letters of
;; the alphabet, or the set of tuples of values in the support of some posterior
;; distribution.
;;
;; A *Markov chain* refers to an ordered sequence of random variables
;; $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \dots$ which have the property
;; that the future is conditionally independent of the past given the current
;; state of the system. In other words
;;
;; $$p(\mathbf{v}_{t+1}, \mathbf{v}_{t+2}, \dots \mid \mathbf{v}_1, \dots, \mathbf{v}_{t})=p(\mathbf{v}_{t+1}, \mathbf{v}_{t+2}, \dots \mid \mathbf{v}_{t})$$
;;
;; The random variables themselves can be discrete, continuous, or represent a
;; compound distribution over a combination of elementary random variables of
;; either or both kinds.
;;
;; A Markov chain can be defined by it's trasition probabilities, which are
;; often represented as a matrix $A$ where
;;
;; $$A_{[\mathbf{v}^\prime, \mathbf{v}]}=p(\mathbf{v}^\prime \mid \mathbf{v})$$
;;
;; when $\mathbf{v}$ is discrete, and or by a probability *kernel* when
;; $\mathbf{v}$ is continuous.
;;
;; $$p(\mathbf{v}^\prime \in B \mid \mathbf{v})$$
;;
;; We will assume that our kernels can be represented by conditional densities.
;;
;; $$p(\mathbf{v}^\prime \mid \mathbf{v})$$
;;
;; In order to fully specify a Markov chain, we must specify the *intial
;; distribution* which is the marginal distribution over $\mathbf{v}_1$.
;;
;; We think about a *run* of a Markov chain as sequence of steps or
;; *transitions* starting from the initial distribution.
;;

;; # The Gibbs Sampler as Markov Chain
;;
;; We can think of our Gibbs sampler as a Markov chain on a state space defined
;; by our target posterior distribution. At each time step $t$ we are in a state
;; $\mathbf{v}_t$ which can be thought of as a trace. We resample one of the
;; latent random variables in the trace, conditional on the rest, and transition
;; to a new state $\mathbf{v}_{t+1}$ which necessarily matches the last state on
;; all but the resampled random choice.
;;
;; What is the transition distribution of the Gibbs sampler $p(\mathbf{v}^\prime
;; \mid \mathbf{v})$?
;;
;; Let's introduce some notation. Let $\mathbf{v}$ be a vector of values for all
;; random variables in our joint distribution. For example, for the GMM,
;;
;; $$\mathbf{v} = \langle \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:t}, x_{1:T} \rangle$$
;;
;; The distribution $p(\mathbf{v})$ is thus the joint distribution of our model.
;;
;; Let $\mathbf{v}_{-z_{i}}$ be a vector of values for all random variables
;; after removing the variable $z_{i}$.
;;
;; $$\mathbf{v}_{-z_{i}} = \langle \phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:(i-1)},
;; z_{(i+1):T}, x_{1:T} \rangle$$
;;
;; The distribution over this vector, $p(\mathbf{v}_{-z_{i}}) = \sum_{z_{i}}
;; p(\mathbf{v})$, is the marginal distribution on the variables marginalizing
;; away $z_{i}$. Note that using the chain rule we can write our joint as
;; $p(z_{i} \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})$.
;;
;; At each step of the Gibbs sampler, we deterministically choose a single $z_i$
;; to resample. Thus, if we start in a state $\mathbf{v}_t$ with joint
;; probability $p(\mathbf{v}_t)$ which can be written as $p(\breve{z}_{i} \mid
;; \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})$. We then resample the value of
;; $z_{i}$ its value from $p(z_{i} \mid \mathbf{v}^t_{-z_{i}})$, to get a new
;; value $\breve{z}^\prime_{i}$. Our new joint state is $\mathbf{v}^\prime$ with
;; $p(\mathbf{v}^\prime)=p(\breve{z}^\prime_{i} \mid
;; \mathbf{v}_{-z_i})p(\mathbf{v}_{-z_i})$.
;;
;; $$\begin{align}
;; p(\mathbf{v}^\prime \mid \mathbf{v}) &= p(z_{i} \mid \mathbf{v}_{-z_{i}}) \\
;;                         &= \frac{p(\mathbf{v})}{p(\mathbf{v}_{-z_{i}})}
;; \end{align}$$
;;
;;

;; # Markov Chain Monte Carlo
;;
;; *Monte Carlo* methods, as we have seen, approximate some expectation (or
;; *distribution) by drawing samples. That is, we design sampling algorithms
;; *with the property that as we draw more and more sample, the histogram of
;; *samples will come to resemble our target posterior more and more closely and
;; *thus any expectation taken with respect to this histogram will tend
;; *approximate any expectation taken with respect to the original distribution
;; *more and more closely.
;;
;; *Markov chain Monte Carlo* methods are the special case of Monte Carlo
;; *methods where we draw samples from a Markov chain&mdash;that is, we count
;; *the number of times that we visit particular states and use this to estimate
;; *our distribution or expectation. In the case of the Gibbs sampler, we
;; *transition from one state in our target posterior to another by resampling a
;; *single latent random variable conditioned on the others.
;;
;;
;; There are a number of differences between this Monte Carlo approximation and
;; the other others we have seen, such as importance sampling. Perhaps the most
;; obvious one is that the states that we visit in MCMC are not at all
;; independent from one another&mdash;each sample only changes a single latent
;; random variable, leaving the others identical.
;;
;; This raises an important question: In what sense can such highly correlated
;; samples from a Markov chain like a Gibbs sampler be said to approximate a
;; target joint posterior?
;;
;; In order to understand the answer to this question, we must first understand
;; the notion of a *stationary distribution*.

;; # Stationary Distributions
;;
;; How do we characterize the distribution that results from running a Markov
;; chain for a long time? The answer this question involves a concept known as
;; the *stationary distribution* of the Markov chain. The stationary
;; distribution is the distribution that measures the proportion of time the
;; Markov chain spends in each state if you run it forever.
;;
;; A distribution $\pi$ is stationary for a Markov chain with transition
;; probabilities $p(\mathbf{v}^\prime \mid \mathbf{v})$ if:
;;
;; $$\pi(\mathbf{v}^\prime) = \sum_{\mathbf{v}} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})$$
;;
;; $$\pi = \pi A$$
;;
;;
;; <!-- $$\rho^\prime(\mathbf{v}_{t+1}=y) = \sum_{x} \rho(\mathbf{v}_t=x)\Pr(\mathbf{v}_{t+1}=y |\mathbf{v}_t=x) \quad \forall y $$ -->
;;
;;
;; <!-- $$\Pr(\mathbf{v}_{t+1}=y) = \sum_{\mathbf{v}} \Pr(\mathbf{v}_{t}=\mathbf{v})\Pr(\mathbf{v}_{t+1}=y |\mathbf{v}_{t}=\mathbf{v}) \quad \forall y $$ -->
;;
;; In other words, if you started with uncertainty over the current state that
;; is characterized by the stationary distribution, and then you compute your
;; updated uncertainty after taking one step in the Markov chain, you end up in
;; the same state of uncertainty given by the stationary distribution.
;;
;; Not all Markov chains have stationary distributions&mdash;some Markov chains
;; have certain kinds of pathological behavior that means that they do not have
;; such a fixed point. For instance, consider the Markov chain on the natural
;; numbers with transition probabilities given by $\pi_{n}(n+1)=1$. However,
;; under some conditions, a Markov chain will have a stationary
;; distribution (positive recurrence), under some additional conditions the
;; stationary distribution will be unique (irreducibility).
;;
;; Most importantly, under some further conditions it can be shown that a Markov
;; chain will converge to its stationary distribution over time (aperiodicity).
;;
;; If we can show that our Markov chain has the right properties (positive
;; recurrence, irreducibility) **and** that the stationary distribution is
;; exactly the target posterior in question, this means we can use the Markov
;; chain as an approximate sampler for our target posterior, just by running it
;; long enough and keeping track of how often it visits each state.
;;
;; In what follows, we will assume that our first three properties hold for our
;; Markov chains and will focus on showing that our target posterior is in fact
;; the stationary distribution of our chains.

;; # The Gibbs Sampler's Stationary Distribution
;;
;; As a reminder, $p(\mathbf{v})$ refers to our joint distribution where
;; $\mathbf{v}$ is a vector with all of the values of every random variable in
;; this joint. For example, for the GMM cluster assignment problem
;;
;; $$p(\mathbf{v}) = p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:T}, x_{1:T})$$.
;;
;; The notation $p(\mathbf{v}_{-z_{i}})$ referring to the marginal distribution
;; over all random variables **except** $z_{i}$, that is,
;;
;; $$p(\mathbf{v}_{-z_{i}}) = p(\phi, \mu_{1:K}, \sigma^2_{1:K},z_{1:(i-1)},z_{(i+1):T}, x_{1:T}) = \sum_{z_i} p(\phi, \mu_{1:K}, \sigma^2_{1:K}, z_{1:T}, x_{1:T})$$.
;;
;; Finally, we refer to the conditional distribution over $z_{i}$ given the rest
;; of the random variables as $p(z_{i} \mid \mathbf{v}_{-z_{i}})$
;;
;; $$p(z_{i} \mid \mathbf{v}_{-z_{i}}) = p(z_{i} | \phi, \mu_{1:K}, \sigma^2_{1:K},z_{1:(i-1)},z_{(i+1):T}, x_{1:T})$$
;;
;; We now show that $p(\mathbf{v})$ is a stationary distribution for the Gibbs
;; update that we gave above.
;;
;; First note  the following identities.
;;
;; $$p(\mathbf{v}) = \frac{p(\mathbf{v})}{p(\mathbf{v}_{-z_i})}p(\mathbf{v}_{-z_i}) = p(z_{i} \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}}) $$
;;
;; Now we can prove the stationarity of our posterior distribution under Gibbs.
;; Let's assume that the value for $z_i$ is $c$ at time step $T$ and $c^\prime$
;; at time step $T+1$.
;;
;; We assume that our start state with $z_i = c$ is distributed as
;; $p(\mathbf{v}_{c})=p(z_{i}=c \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})$
;; and we wish to show that after taking a step using our Gibbs update to $z_i = c^\prime$,
;; the marginal probability distribution over states is also $p$
;;
;; $$\begin{align}
;; \Pr(\mathbf{v}_{c^\prime}) &=  \sum_{c} p(\mathbf{v}_c) \times p(z_{i}=c^\prime \mid \mathbf{v}_{-z_{i}})\\
;; \ &=  \sum_{c} [p(z_{i}=c \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})] \times p(z_{i}=c^\prime \mid \mathbf{v}_{-z_{i}})\\
;;  &=  \sum_{c} p(z_{i} = c  \mid \mathbf{v}_{-z_{i}}) \times [p(\mathbf{v}_{-z_{i}})  p(z_{i} = c^\prime \mid \mathbf{v}_{-z_{i}})]\\
;;  &=  \sum_{c} p(z_{i} = c  \mid \mathbf{v}_{-z_{i}})  p(\mathbf{v}_{c^\prime}) \\
;;   &=  p(\mathbf{v}_{c^\prime})\sum_{c} p(z_{i} = c  \mid \mathbf{v}_{-z_{i}})   \\
;;  &=  p(\mathbf{v}_{c^\prime})
;; \end{align}$$
;;
;; Thus, after taking a step using our Gibbs distribution, our distribution over
;; states is still $p$ if we started from the distribution $p$.
;;
;; Note that any target posterior we are interested in involves just a subset of
;; the variables from the joint (for example $z_{1:T}$). The posterior
;; probability of this set of target random variables in posterior space is
;; proportional to the joint probability of those target random variables. Since
;; the Gibbs sampler fixes and does not resample the other variables, and since
;; sampling works up to a normalizing constant, this shows that the posterior is
;; also a stationary distribution for the Gibbs sampler.
;;
;; <!-- todo: can I be more precise with what i said above -->
;;
;; If we can show that these updates will also hit every state in the posterior
;; with positive probability (irreducibility) and do not have any
;; loops (aperiodicity) then we know that our Markov chain will converge to the
;; target posterior distribution.

;; # Gibbs Sampling the Means and Variances
;;
;; We have seen how we can define a Gibbs sampler for the component assigment
;; variables in our GMM model, but what about the means and variances of the
;; component distribution? Of course, since the Gibbs sampler involves exactly
;; enumerating and scoring all of the possible choices for the target random
;; variable, we will not be able to Gibbs sample a continuous random variable
;; unless there is a closed form solution to the conditional posterior density
;; for the variable in question. In the case of this model, it is possible to
;; derive a closed-form solution for the relevant conditional posteriors.
;;
;; In fact, in both the cases of the component means with normal priors and the
;; component variances with inverse gamma priors, the posteriors are also normal
;; and inverse gamma, respectively. This is an instance of the important
;; statistical concept of *conjugacy*. A pair of distributions representing a
;; prior and likelihood is *conjugate* if the posterior distribution computed by
;; renormalizing their product is also of the same parametric form as the prior
;; distribution.
;;
;; Here we simply state the main results for the means $\mu$ and variances
;; $\sigma^2$.
;;
;; The conditonal posterior on the means is given by:
;;
;; $$\begin{align}
;; p(\mu_j \mid \phi, \mu_{-j}, \sigma^2_{1:K}, z_{1:T}, x_{1:T}) &= p(\mu_j \mid \sigma^2_{j}, x_{\{t: z_t=j\}})
;; \end{align}$$
;;
;; Because of conjugacy, the posterior distribution on $\mu_j$ is also a normal
;; distribution with parameters $\hat\mu_j$ and $\hat\sigma^2_j$.
;;
;; $$\begin{align}
;; \mu_j\mid x_{\{t: z_t=j\}}, \sigma^2_{j} \sim \mathrm{normal}(\hat\mu_j, \hat\sigma^2_j)
;; \end{align}$$
;;
;; The values for these updated parameters are given by:
;;
;; $$\begin{align}
;; \hat\mu_j = \frac{\kappa\xi + \frac{1}{\sigma_j^2} \sum_{\{t: z_t=j\}} x_t}{\kappa + 1/\sigma^2_j}
;; \qquad
;; \hat\sigma^2_j = \frac{1}{\kappa + n_j/\sigma_j^2}
;; \end{align}$$
;;
;; The conditional posterior of the variances is given by
;;
;; $$\begin{align}
;; p(\sigma^2_j \mid \phi, \mu_{1:K}, \sigma^2_{-j}, x_{1:T}, z_{1:T}) &= \Pr(\sigma^2_j \mid \mu_{j}, x_{\{t: z_t=j\}})
;; \end{align}$$
;;
;; Because of conjugacy, the posterior distribution on $\sigma^2_j$ is also an
;; inverse gamma distribution with parameters $\hat\alpha$ and $\hat\beta$.
;;
;; $$\begin{align}
;; \sigma^2_{j}\mid x_{\{t: z_t=j\}}, \mu_j  \sim \mathrm{InverseGamma}(\hat\alpha, \hat\beta)
;; \end{align}$$
;;
;; The values for these updated parameters are given by:
;;
;; $$\begin{align}
;; \hat\alpha = \alpha + \frac{T}{2}
;; \qquad
;; \hat\beta = \beta + \frac{1}{2}\sum_{\{t: z_t=j\}} \mu_j - x_t
;; \end{align}$$
;;
;; # The Metropolis-Hastings Algorithm
;;
;; In Gibbs sampling, we sampled our proposed change to variable $z_i$ from the
;; full conditional posterior distribution over this variable given everything
;; else. When it is possible to efficiently and exactly compute this
;; distribution&mdash;either by enumeration or in closed form&mdash;Gibbs
;; sampling can be a very effective approach to MCMC. However, like the full
;; posterior, even the conditional posterior can be impossible or inefficient to
;; compute. What do we do then?
;;
;; As with importance sampling, one alternative idea is to sample from a
;; *proposal distribution* $q(\mathbf{v}^\prime \mid \mathbf{v})$. Similarly to
;; rejection sampling, we will correct this proposal to the true distribution by
;; *accepting* or *rejecting* the proposed trace. One particular way of
;; accepting or rejection proposals gives rise to the *Metropolis-Hastings
;; Algorithm* often known as *MH*. In Gen, this is a built-in algorithm
;; implemented using the `Gen.mh` function. Let's see how this works.
;;
;; First, let's consider a different inference problem where we resample the
;; $\mu$s and $\sigma^2$s while leaving all other latents fixed. First we
;; implement a new version of `get_constrained_trace` which captures this idea.

;; ```julia
;; function get_constrained_trace(true_trace)
;;     K, T = get_args(true_trace)

;;     observations = Gen.choicemap()

;;     observations[:ϕ] = true_trace[:ϕ]


;;     for i in 1:T
;;         observations[:z => i] = true_trace[:z => i]
;;         observations[:x => i] = true_trace[:x => i]
;;     end

;;     (t,w) = Gen.generate(gmm, (K,T), observations)
;; end


;; constrained_trace,w=get_constrained_trace(true_trace)
;; render_gmm_trace(constrained_trace)
;; ```


;; Next, we write our proposal. Proposal distributions in MCMC are just
;; conditional distributions that resample our target variables and thus we
;; implement them using generative functions in Gen. In order to be used by
;; `Gen.mh`, the proposal distribution must take the last state in the Markov
;; chain as it's first argument and it must resample any random variables it
;; wants to change, giving them the same name as in the original generative
;; process.
;;
;; We will implement a proposal distribution for our $\mu$s and $\sigma^2$s
;; known as a *Gaussian drift* proposal, which just means sampling a new value
;; for each mean and variance from a Gaussian centered on the old value. Since
;; variances must be positive, we will use a truncated Gaussian that is only
;; positive as our drift kernel to sample new variances.
;;

;; ```julia
;; @gen function gmm_drift_proposal(trace)

;;     K, T = get_args(trace)

;;     μs= Vector{Float64}(undef, K)
;;     σ²s=Vector{Float64}(undef, K)
;;     for j=1:K
;;         μs[j]= {:μ => j} ~  normal(trace[:μ => j], 10)
;;         σ²s[j]= {:σ² => j} ~ trunc_normal(trace[:σ² => j], 10)
;;     end
;; end

;; num_steps=10000
;; t = constrained_trace
;; accepted=Vector{Bool}(undef, num_steps)
;; for sweep=1:num_steps
;;   (t, accepted[sweep]) = Gen.mh(t, gmm_drift_proposal, ())
;; end

;; print("Percentage of accepted transitions:" *
;;        string(sum(accepted)/num_steps))
;; ```

;; ```julia
;; render_gmm_trace(constrained_trace)

;; render_gmm_trace(t)
;; ```

;; Here we plugged our proposal into `Gen.mh` which takes three arguments
;;
;;  1. The trace representing the previous state.
;;  2. The proposal distribution represented as a generative function.
;;  3. Any arguments to be passed to the proposal distribution in addition to the trace representing the last state in the Markov chain.
;;
;; `Gen.mh` returns a new trace, as well as a binary indicator variable
;; representing whether or not the new trace is the result of accepting the
;; proposal, or is simply the same trace as before (i.e., that the proposal was
;; rejected).
;;
;; The proportion of accepted traces is an important diagnostic for the MH algorithm and thus we display this information as well.

;; > What happens if we increase or decrease the variances on the proposal normal distributions?

;; # The Metropolis Adjusted Langevin Algorithm
;;
;; One major advantage of the MH algorithm is that we can use whatever proposal
;; we like, including ones that take better advantage of the structure of the
;; model. One improved approach to the problem above is the *Metropolis adjusted
;; Langevin* algorithm where we differentiate the log score of original trace
;; with respect to the variables we wish to resample and then make use of the
;; gradients in order to sample values in the right direction.

;; ```julia
;; @gen function gmm_mala_proposal(trace, tau)

;;     K, T = get_args(trace)

;;     selection=select([:μ => j for j=1:K]...)
;;     (arg_grads, choice_values, choice_grads) =
;;                   choice_gradients(trace, selection, nothing)

;;     μs= Vector{Real}(undef, K)
;;     σ²s=Vector{Real}(undef, K)
;;     for j=1:K
;;          #println(choice_grads[:μ => j])

;;         μs[j]= {:μ => j} ~  normal(choice_values[:μ => j]+tau*choice_grads[:μ => j],
;;                                    sqrt(2 * tau))
;;         σ²s[j]= {:σ² => j} ~  trunc_normal(trace[:σ² => j], sqrt(2 * tau))
;;     end
;; end

;; num_steps=1000
;; t = constrained_trace
;; accepted=Vector{Bool}(undef, num_steps)
;; for sweep=1:num_steps
;;   (t, accepted[sweep]) = Gen.mh(t, gmm_mala_proposal, (0.01,))
;; end

;; print("Percentage of accepted transitions:" *
;;        string(sum(accepted)/num_steps))



;; ```

;; ```julia
;; render_gmm_trace(constrained_trace)

;; render_gmm_trace(t)
;; ```

;; # The MH Acceptance Ratio
;;
;; How does the accept/reject step for each transition in MH work?
;;
;; Assume that our target posterior distribution is given by $p$ (which we may
;; only know up to a normalizing constant, i.e., we know the unnormalized joint
;; distribution $p^*$). In MH we sample from a proposal transition distribution
;; $q(\mathbf{v}^\prime \mid \mathbf{v})$. We then compute the so-called MH
;; ratio:
;;
;; $$R_{\mathrm{MH}}=\frac{p(\mathbf{v}^\prime)}{p(\mathbf{v})}\frac{q(\mathbf{v} \mid \mathbf{v}^\prime)}{q(\mathbf{v}^\prime \mid \mathbf{v})}$$
;;
;; This ratio involves four quantities $p(\mathbf{v}^\prime)$ the score of the
;; proposed state under our model, $p(\mathbf{v})$ the score of the original
;; state under our model, $q(\mathbf{v}^\prime\mid \mathbf{v})$ the proposal
;; probability of state $\mathbf{v}^\prime$ and $q(\mathbf{v}\mid
;; \mathbf{v}^\prime)$ the *backwards probability* of the original state,
;; starting from the proposed state. We use these quantities to compute the *MH
;; acceptance threshold*.
;;
;; $$\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}) = \mathrm{min}\left(1,  \frac{p(\mathbf{v}^\prime)}{p(\mathbf{v})}\frac{q(\mathbf{v}_{t} \mid \mathbf{v}^\prime)}{q(\mathbf{v}^\prime \mid \mathbf{v})}\right)$$
;;
;; In other words, when $R_{\mathrm{MH}}$ is greater than $1$ we set $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v})$ to $1$ and when $0<R_{\mathrm{MH}}<1$ we leave it as is.
;;
;; We then flip a coin with weight $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v})$
;; and *accept* the proposed move if it comes up heads and *reject* the proposed
;; move if it comes up tails. In the latter case, we transition back to our
;; original state $\mathbf{v}$ (or, equivalently, remain in it).
;;
;; Why does this lead to a correct MCMC algorithm? To understand this, we will
;; first need to introduce several new ideas: *balance* and *detailed balance*.
;;
;; <!-- TODO: decide what notation you want to use for current and last states, prime or t/t+1, or v_c and v_cprime -->

;; # Balance and Detailed Balance
;;
;; One intuitive implication of the definition of a stationary distribution is
;; that it gives rise to a property of the Markov chain which we'll call the
;; *global balance property*. Global balance with respect to a stationary
;; distribution means that the amount of *flux*&mdash;the probability mass moved
;; into or out of a state by a transition kernel&mdash;must be equal when you
;; are in a stationary distribution.
;;
;; $$\begin{align}
;; \pi(\mathbf{v}^\prime) &= \sum_{\mathbf{v}}  \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})\\
;; \pi(\mathbf{v}^\prime) &= \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v}) +  \pi(\mathbf{v}^\prime)p(\mathbf{v}^\prime |\mathbf{v}^\prime)\\
;; \pi(\mathbf{v}^\prime) - \pi(\mathbf{v}^\prime)p(\mathbf{v}^\prime |\mathbf{v}^\prime) &= \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})\\
;; \pi(\mathbf{v}^\prime)\left[ 1 - p(\mathbf{v}^\prime |\mathbf{v}^\prime) \right] &= \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})\\
;; \pi(\mathbf{v}^\prime)\sum_{\mathbf{v} \neq \mathbf{v}^\prime} p(\mathbf{v} |\mathbf{v}^\prime) &= \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})\\
;; \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v}^\prime)p(\mathbf{v} |\mathbf{v}^\prime) &= \sum_{\mathbf{v} \neq \mathbf{v}^\prime} \pi(\mathbf{v})p(\mathbf{v}^\prime |\mathbf{v})\\
;; \end{align}$$
;;
;; The global balance property is a property associated with a Markov chain
;; defined on some state space $\Pr(\mathbf{v}_{t+1} \mid \mathbf{v}_{t})$ with
;; respect to a distribution over the same state space $\pi$. If the Markov
;; chain satisfies thus property with respect to $\pi$, then $\pi$ must be a
;; stationary distribution for the Markov chain. Thus, if we fix some $\pi$ and
;; we can find a Markov chain that satisfies global balance, then we have found
;; a Markov chain for which $\pi$ is stationary.
;;
;; In practice, we make use of a related, but stronger property than global
;; balance known as the *detailed balance condition*.
;;
;; A Markov chain satisfies *detailed balance* (or is called *reversible*) for
;; some distribution $\pi$ if
;;
;; $$\pi(\mathbf{v})p(\mathbf{v}^\prime \mid \mathbf{v}) = \pi(\mathbf{v}^\prime)p(\mathbf{v} \mid \mathbf{v}^\prime), \forall \mathbf{v}, \mathbf{v}^\prime$$
;;
;; Detailed balance clearly implies global balance: if the total probability
;; transfered from $x$ to $y$ is equal to the the total probability transferred
;; in the other direction, it must be the case that ingoing and outgoing flux
;; are equal. In fact, detailed balance is a stronger condition than global
;; balance: the latter can be satisfied even if the former cannot, but detailed
;; balance is sufficient for global balance.
;;
;; This also shows that if a Markov chain satisfies detailed balance for some
;; marginal distribution on states $\pi$ then $\pi$ is a stationary distribution
;; of the chain.
;;
;; Thus, if we can construct a Markov chain that satisfies detailed balance for
;;our target posterior, it's stationary distribution will be our target
;;posterior.

;; # MH Satisfies Detailed Balance
;;
;; Why does the MH accept/reject process give us a Markov chain whose stationary
;; distribution is the target posterior $\pi(\mathbf{v})$?
;;
;; We can prove that the stationary distribution of this Markov process is the
;; correct posterior by showing that it satisfies detailed balance for the
;; target posterior $\pi$.
;;
;; First, consider the transition probability of our MH chain. We transition
;; from state $\mathbf{v}$ to state $\mathbf{v}^\prime$ with probability
;; $q(\mathbf{v}^\prime \mid
;; \mathbf{v}_{t})\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t})$. Thus, we can
;; show that detailed balance is satisfied if the following equation holds.
;;
;; $$\pi(\mathbf{v}_{t})q(\mathbf{v}^\prime \mid \mathbf{v}_{t})\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t})=\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime)\mathcal{A}(\mathbf{v}_{t},\mathbf{v}^\prime)$$
;;
;; Since $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t})$ and
;; $\mathcal{A}(\mathbf{v}_{t},\mathbf{v}^\prime)$ are reciprocals of one
;; another, there are three cases. First, if
;; $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t})=\mathcal{A}(\mathbf{v}_{t},\mathbf{v}^\prime)$
;; that implies that the ratio is $1$ and thus the detailed balance equation
;; must hold.
;;
;; Second, if $\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime) >
;; \pi(\mathbf{v}_{t})q(\mathbf{v}^\prime \mid \mathbf{v}_{t})$ then
;; $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t}) = 1$, and
;; $\mathcal{A}(\mathbf{v}_{t},\mathbf{v}^\prime) =
;; \frac{\pi(\mathbf{v}_{t})q(\mathbf{v}^\prime \mid
;; \mathbf{v}_{t})}{\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t}\mid
;; \mathbf{v}^\prime)}$ which is betwen $0$ and $1$.
;;
;; In this case, our detailed balance equation becomes
;;
;; $$\begin{align}\pi(\mathbf{v})q(\mathbf{v}^\prime \mid \mathbf{v}_{t}) \times 1 &= \pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime) \times \frac{\pi(\mathbf{v}_{t})}{\pi(\mathbf{v}^\prime)}\frac{q(\mathbf{v}^\prime \mid \mathbf{v}_{t})}{q(\mathbf{v}_{t}\mid \mathbf{v}^\prime)}\\
;; \pi(\mathbf{v})q(\mathbf{v}^\prime \mid \mathbf{v}_{t}) &= \pi(\mathbf{v})q(\mathbf{v}^\prime \mid \mathbf{v}_{t})\\
;; \end{align}$$
;;
;;
;; Third and finally, if $\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid
;; \mathbf{v}^\prime) < \pi(\mathbf{v}_{t})q(\mathbf{v}^\prime \mid
;; \mathbf{v}_{t})$ then $\mathcal{A}(\mathbf{v}_{t},\mathbf{v}^\prime) = 1$,
;; and $\mathcal{A}(\mathbf{v}^\prime,\mathbf{v}_{t}) =
;; \frac{\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid
;; \mathbf{v}^\prime)}{\pi(\mathbf{v}_{t})q(\mathbf{v}^\prime\mid
;; \mathbf{v}_{t})}$ which is betwen $0$ and $1$.
;;
;;
;; $$\begin{align}\pi(\mathbf{v})q(\mathbf{v}^\prime \mid \mathbf{v}_{t}) \times \frac{\pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime)}{\pi(\mathbf{v}_{t})q(\mathbf{v}^\prime\mid \mathbf{v}_{t})} &= \pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime) \times 1\\
;; \pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime) &= \pi(\mathbf{v}^\prime)q(\mathbf{v}_{t} \mid \mathbf{v}^\prime)\\
;; \end{align}$$
;;
;; Thus, our MH kernel satisfies detailed balance for $\pi$ and the resulting
;; Markov chain has $\pi$ as its stationary distribution.

;; # Gibbs Sampling and MH
;;
;; It turns out that the Gibbs sampler is a special case of MH.
;;
;; $$\begin{align}R_{\mathrm{MH}}&=\frac{p(\mathbf{v}^\prime)}{p(\mathbf{v})}\frac{q(\mathbf{v} \mid \mathbf{v}^\prime)}{q(\mathbf{v}^\prime\mid \mathbf{v})}\\
;; R_{\mathrm{MH}}&=\frac{p(z_{i}=c^\prime \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})}{p(z_{i}=c \mid \mathbf{v}_{-z_{i}})p(\mathbf{v}_{-z_{i}})}\frac{p(z_{i}=c \mid \mathbf{v}_{-z_{i}})}{p(z_{i}=c^\prime \mid \mathbf{v}_{-z_{i}})}\\
;; R_{\mathrm{MH}}&=\frac{p(\mathbf{v}_{-z_{i}})}{p(\mathbf{v}_{-z_{i}})}\\
;; R_{\mathrm{MH}}&=1\\
;; \end{align}$$
;;
;; In other words, the Gibbs sampler always accepts.
;;

;; # The Metropolis Algorithm
;;
;; When the forward and reverse moves have the same probability, the proposal is
;; called *symmetric* and the relevant portion of the MH acceptance ratio
;; cancels. This was the case with our Gaussian drift proposal for the
;; means (but not the variances, why?).
;;
;; The resulting algorithm with symmetric proposals is called the Metropolis
;; algorithm.
;;
;; $$\begin{align}R_{\mathrm{M}}&=\frac{p(\mathbf{v}^\prime)}{p(\mathbf{v})}
;; \end{align}$$

;; # Composing MCMC Kernels
;;
;; We can interleave kernels of any type.
;;

;; ```julia
;; function get_constrained_trace(true_trace)
;;     K, T = get_args(true_trace)

;;     observations = Gen.choicemap()

;;     observations[:ϕ] = true_trace[:ϕ]

;;     for i in 1:T
;;         observations[:x => i] = true_trace[:x => i]
;;     end

;;     (t,w) = Gen.generate(gmm, (K,T), observations)
;; end

;; num_steps=10000
;; constrained_trace,w=get_constrained_trace(true_trace)
;; t = constrained_trace
;; accepted=Vector{Bool}(undef, num_steps)
;; for sweep=1:num_steps
;;   (t, accepted[sweep]) = Gen.mh(t, gmm_mala_proposal, (0.01,))
;;   t=gmm_gibbs_sweep(t, :z)
;; end

;; print("Percentage of accepted transitions:" *
;;        string(sum(accepted)/num_steps))
;; ```

;; ```julia
;; render_mixture_trace(constrained_trace);
;; render_mixture_trace(t)
;; ```
