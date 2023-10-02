;; In the last portion of the course, we looked at handling models with complex
;; state spaces using MCMC. Now we turn to another approach to models with
;; complex state spaces: *sequential monte carlo* (SMC). In MCMC we sample from
;; our target distribution by iteratively adjusting subparts of our joint state.
;; In SMC, we instead assume that the random variables in our joint distribution
;; states are **ordered** and we sample them sequentially with later samples
;; conditioned on the earlier samples. SMC techniques were originally developed
;; for time series data and it is most natural to explain them in this setting.
;; However, it should be kept in mind that we are free to impose an ordering on
;; the random variables in any model and thus SMC can be used as an alternative
;; to MCMC (or in combination with MCMC) for sampling from distributions with
;; complex joint states.
;;
;;

;; ```julia
;; using Gen
;; using Distributions
;; using Plots
;; using Printf

;; import Logging; Logging.disable_logging(Logging.Info) # Disable info messages when making gif
;; ```

;; To introduce SMC, we will use the Hidden Markov model as our running example.
;; Unlike the defition we used earlier in the course, however, we will fix our
;; initial, transition, and observation distributions.

;; ```julia
;; T=100
;; K=50
;; O=10

;; δ = fill(10.0,K)
;; α = fill(10.0,K)
;; ϵ = fill(0.1,O)

;; ;; initial distribution over latent states
;; ϕ = rand(Dirichlet(δ))

;; transition = zeros(K,K)
;; for k in 1:K
;;     transition[k,:] = rand(Dirichlet(α))
;; end

;; observation = zeros(K,O)
;; for k in 1:K
;;     observation[k,:] =  rand(Dirichlet(ϵ))
;; end

;; @gen function hmm(T, ϕ, transition, observation)

;;     xs=Vector{Int}(undef, T)
;;     zs=Vector{Int}(undef, T)

;;     zs[1] = {:z => 1} ~ categorical(ϕ)
;;     xs[1] = {:x => 1} ~ categorical(observation[zs[1],:])

;;     for t in 2:T
;;         zs[t] = {:z => t} ~ categorical(transition[zs[t-1],:])
;;         xs[t] = {:x => t} ~ categorical(observation[zs[t],:])
;;     end
;;     return(xs)
;; end

;; ground_truth=Gen.simulate(hmm, (T, ϕ, transition, observation));
;; get_choices(ground_truth)
;; ```

;; Similar to our study of the GMM model in the unit on MCMC, we will focus on
;; the posterior distribution over our state-assigment variables $z_t$.
;;
;; $$p(z_{1:t} | x_{1:t}, \phi, A, E)$$
;;
;; To simplify the notation, let's suppress the hyperparameters and use the
;; probability mass function $p$ to refer to this distribution.
;;
;; $$p(z_{1:t} | x_{1:t})$$
;;
;; Our joint distribution is given by the following expression.
;;
;; $$p(z_{1:t}, x_{1:t}) = \phi(z_1)p(x_1 \mid z_1)\prod_{s=2}^t p(z_s \mid z_{s-1})p(x_s \mid z_{s})$$
;;
;; This joint posterior is defined using conditioning as
;;
;; $$\begin{align}
;; p(z_{1:t} | x_{1:t}) &= \frac{p(z_{1:t}, x_{1:t})}{p(x_{1:t})}\\
;; p(z_{1:t} | x_{1:t})&= \frac{p(z_{1:t}, x_{1:t})}{ \sum_{z_{1:t}} p(z_{1:t}, x_{1:t})}\\
;; p(z_{1:t} | x_{1:t})&=\frac{\phi(z_1)p(x_1 \mid z_1)\prod_{s=2}^t p(z_s \mid z_{s-1})p(x_s \mid z_{s})}{\sum_{z_{1:t}} \phi(z_1)p(x_1 \mid z_1)\prod_{s=2}^t p(z_s \mid z_{s-1})p(x_s \mid z_{s})}\end{align}$$
;;
;; In the case of the HMM model, the marginal on the bottom of this fraction can
;; be computed efficiently using dynamic programming (using so-called forward
;; algorithm); alternatively, we can also define a Gibbs sampler that samples
;; from the conditional posterior distributions $p(z_{i} | z_{1:(i-1)},
;; z_{(i+1):t}, x_{1:t})$.
;;
;; However, in this unit we will take a different approach: *Sequential Monte
;; Carlo*.
;;

;; # Sequential Monte Carlo: Sequentializing the HMM Posterior
;;
;; The idea behind SMC is that we would like to update our model sequentially
;; incoporating sampling the $z_i$ in order $1 \dots t$, and incorporating the
;; previous samples as well as the data $x_{1:t}$ into the distribution that we
;; use for subsequent $z_i$'s.
;;
;; This suggests that we might use the chain rule decomposition of the posterior
;; to derive a sequence of conditional distributions over states which can be
;; sampled one at a time, in order.
;;
;; Consider our posteriori distribution $p(z_{1:t} | x_{1:t})$. Using the chain
;; rule, we can rewrite this into a sequence of conditional distributions:
;;
;; $$ \begin{align}p(z_{1:t} | x_{1:t}) &= p(z_{1} | x_{1:t})p(z_{2} | z_{1}, x_{1:t})p(z_{3} | z_{1:2}, x_{1:t})\dots p(z_{t} | z_{1:(t-1)}, x_{1:t}) \\
;;  &= p(z_1 \mid x_{1:t})\prod_{s=2}^t p(z_s \mid z_{1:(s-1)}, x_{1:t}) \end{align}$$
;;
;; The first conditional, $p(z_1 \mid x_{1:t})$ can be computed using Bayes'
;; rule
;;
;; $$\begin{align}p(z_1 \mid x_{1:t})
;; &= \frac{ p(z_1, x_{1:t})}{\sum_{z^\prime_1} p(z^\prime_1, x_{1:t})}\\
;; &= \frac{ \sum_{z_{2:t}} p(z_1, x_1, z_{2:t}, x_{2:t})}{\sum_{z^\prime_1} \sum_{z^\prime_{2:t}} p(z^\prime_1, x_1, z^\prime_{2:t}, x_{2:t})}\\
;; &= \frac{\sum_{z_{2:t}} \phi(z_1)p(x_{1}\mid z_1) \prod_{s=2}^t p(z_{s} \mid z_{s-1}) p(x_{s} \mid z_{s})}{\sum_{z^\prime_1} \sum_{z^\prime_{2:t}} \phi(z_1)p(x_{1}\mid z^\prime_1) \prod_{s=2}^t p(z^\prime_{s} \mid z^\prime_{s-1}) p(x_{s} \mid z^\prime_{s})}\\
;; &= \frac{ \phi(z_1)p(x_{1}\mid z_1) \sum_{z_{2:t}} \prod_{s=2}^t p(z_{s} \mid z_{s-1}) p(x_{s} \mid z_{s})}{\sum_{z^\prime_1}  \phi(z_1)p(x_{1}\mid z^\prime_1) \sum_{z^\prime_{2:t}}\prod_{s=2}^t p(z^\prime_{s} \mid z^\prime_{s-1}) p(x_{s} \mid z^\prime_{s})}\\
;; &= \frac{ \phi(z_1)p(x_{1}\mid z_1) \sum_{z_{2:t}} p(z_{2:t}, x_{2:t} \mid z_1)}{
;; \sum_{z^\prime_1}  \phi(z^\prime_1)p(x_{1}\mid z^\prime_1) \sum_{z^\prime_{2:t}}  p(z^\prime_{2:t}, x_{2:t} \mid z^\prime_1)}\\
;; &= \frac{ \phi(z_1)p(x_{1}\mid z_1)  p(x_{2:t} \mid z_1)}{
;; \sum_{z^\prime_1}  \phi(z^\prime_1)p(x_{1}\mid z^\prime_1)  p(x_{2:t} \mid z^\prime_1)}\\
;; \end{align}$$
;;
;; Subsequent conditionals can also be computed in a similar way. Assuming that
;; $z_{s-1}$ is distributed according to its true posterior, we can recursively
;; sample
;;
;; $$ \begin{align}p(z_s \mid z_{1:(s-1)}, x_{1:t})
;; &= \frac{\phi(z_1)p(x_1 \mid z_1)\prod_{i=2}^{(s-1)} p(z_i \mid z_{i-1})p(x_i \mid z_{i})\times\quad~~ p(z_s \mid z_{(s-1)})p(x_s \mid z_{s}) \sum_{z_{(s+1):t}}p(z_{(s+1):t}, x_{(s+1):t} \mid z_{s})}
;;         {\phi(z_1)p(x_1 \mid z_1)\prod_{i=2}^{(s-1)} p(z_i \mid z_{i-1})p(x_i \mid z_{i})\times \sum_{z^\prime_s} p(z^\prime_s \mid z_{(s-1)})p(x_s \mid z^\prime_{s})\sum_{z^\prime_{(s+1):t}}p(z^\prime_{(s+1):t}, x_{(s+1):t} \mid z^\prime_s)}\\
;;  &= \frac{p(z_s \mid z_{(s-1)})p(x_s \mid z_{s})\sum_{z_{(s+1):t}}p(z_{(s+1):t}, x_{(s+1):t} \mid z_s)}
;;         { \sum_{z^\prime_s} p(z^\prime_s \mid z_{(s-1)})p(x_s \mid z^\prime_{s})\sum_{z^\prime_{(s+1):t}}p(z^\prime_{(s+1):t}, x_{(s+1):t} \mid z^\prime_s)}\\
;; &= \frac{p(z_s \mid z_{s-1})p(x_s \mid z_{s})p(x_{(s+1):t} \mid z_s)}
;;         { \sum_{z^\prime_s} p(z^\prime_s \mid z_{s-1})p(x_s \mid z^\prime_{s})p(x_{(s+1):t} \mid z^\prime_s) }\\
;;      &= p(z_s \mid z_{s-1}, x_{s:t})
;; \end{align}$$
;;
;; The last line shows that in fact, if we know the state at the preceding time
;; step $z_{s-1}$, then we can forget the rest of the past. This simplifies our
;; chain rule decomposition somewhat.
;;
;; $$ \begin{align}p(z_{1:t} | x_{1:t}) &= p(z_{1} | x_{1:t})p(z_{2} | z_{1}, x_{2:t})p(z_{3} | z_{2}, x_{3:t})\dots p(z_{t} | z_{t-1}, x_{t}) \\
;;  &= p(z_1 \mid x_{1:t})\prod_{s=2}^t p(z_s \mid z_{s-1}, x_{s:t}) \end{align}$$
;;
;; To use this sequence of distributions as a sampler, we could draw samples for
;; each $z_s$ in order, starting from the first distribution and moving forward.
;; Once we reach the end of our observed sequence, we will have drawn a sample
;; from our target posterior: $p(z_{1:t} | x_{1:t})$.
;;
;; However, to compute this sequence of exact componenent-wise posterior
;; distributions over each state $z_s$, we have to be able to efficient compute
;; $p(x_{(s+1):t} \mid z_s)$, **the conditional probability of remainder of the
;; string given the target state $p(x_{(s+1):t} \mid z_s)$**. This quantity is
;; known as the *backwards probability* in the theory of HMMs, and since it
;; involves a sum over all possible state sequences over the suffix of our
;; string, it can be costly to compute.
;;
;; <!-- These conditional distributions are known as the *locally optimal
;; sequential Monte Carlo proposals* for reasons we will make clear below. -->
;;
;; <!-- We can derive this same result by expressing the posterior as a
;; recurrence as follows (note that we continue to suppress $\phi, A, E$ for
;; clarity here):
;;
;; $$\begin{align}
;; p(z_{1:t} \mid x_{1:t}) &= \frac{p(z_{1:t}, x_{1:t})}{ p(x_{1:t})} \\
;; p(z_{1:t} \mid x_{1:t}) &= \frac{p(z_{1:t-1}, x_{1:t-1})p(z_{t} \mid z_{t-1})p(x_{t}\mid z_{t})}{ p(x_{1:t})} \\
;; p(z_{1:t} \mid x_{1:t}) &= \frac{p(z_{1:t-1}, x_{1:t-1})p(z_{t} \mid z_{t-1})p(x_{t}\mid z_{t})}{ p(x_{1:t-1})p(x_{t} \mid x_{1:t-1})} \\
;; p(z_{1:t} \mid x_{1:t}) &= \frac{p(z_{1:t-1}, x_{1:t-1})}{p(x_{1:t-1})} \frac{p(z_{t} \mid z_{t-1})p(x_{t}\mid z_{t})}{p(x_{t} \mid x_{1:t-1})} \\
;; p(z_{1:t} \mid x_{1:t}) &= p(z_{1:t-1} \mid x_{1:t-1}) \frac{p(z_{t} \mid z_{t-1})p(x_{t}\mid z_{t})}{p(x_{t} \mid x_{1:t-1})} \\
;; \end{align}
;; $$ -->
;;
;; <!--
;; Note that the normalizing constant on the final term $p(x_{t} \mid x_{1:t-1})$ is equal to the normalizing constant we derived above.
;;
;; $$\begin{align}p(x_{t} \mid x_{1:t-1}) &= \frac{p(x_1, \dots, x_t)}{p(x_{1}, \dots, x_{t-1})} \\
;; &=\frac{p(x_{1:t})}{p(x_{1:t-1})} \\
;; &= \frac{\sum_{z_{1:t}} p(z_{1:t}, x_{1:t})}{\sum_{z_{1:(t-1)}}  p(z_{1:t-1}, x_{1:t-1})}\\
;; &= \frac{\sum_{z_1} \phi(z_1)p(x_1 \mid z_1) \sum_{z_2} p(z_2 \mid z_1)p(x_2 \mid z_2) \dots \sum_{z_{(t-1)}} p(z_{(t-1)} \mid z_{(t-2)})p(x_{(t-1)} \mid z_{(t-1)})\sum_{z_t} p(z_t \mid z_{(t-1)})p(x_t \mid z_t)}
;; {\sum_{z_1} \phi(z_1)p(x_1 \mid z_1) \sum_{z_2} p(z_2 \mid z_1)p(x_2 \mid z_2) \dots \sum_{z_{(t-1)}} p(z_{(t-1)} \mid z_{(t-2)})p(x_{(t-1)} \mid z_{(t-1)})~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~}\\
;; &= \sum_{z_t} p(z_t \mid z_{(t-1)})p(x_t \mid z_t)\\
;; &= \sum_{z_t} p(x_t, z_t \mid z_{(t-1)})\\
;; &= p(x_t \mid z_{(t-1)})\\
;; \end{align}$$ -->
;;
;; <!--&= \sum_{z_t} p(x_t, z_t \mid z_{(t-1)})\\
;; &= p(x_t \mid z_{(t-1)})\\ -->
;;
;; By writing our posterior using this decomposition, we have written it a a
;; series of single-random-variable conditional distribution, which can be
;; sampled in sequence. By sampling each $z_s$ in this sequence in turn, we can
;; draw a single sample from our posterior $p(z_{1:t} \mid x_{1:t})$. However,
;; we saw the conditionals over $z_s$ are costly to compute and to some extent
;; this defeats the whole idea of the approach. We will need to take another
;; approach.

;; # Locally Optimal Sequential Sampling
;;
;; We saw that the sequntial decomposition of our posterior using the chain rule
;; led to a series of local distributions,
;;
;; $$p(z_s \mid z_{s-1}, x_{s:t}) = \frac{p(z_s \mid z_{s-1})p(x_s \mid z_{s})p(x_{(s+1):t} \mid z_s)}
;;         { \sum_{z^\prime_s} p(z^\prime_s \mid z_{s-1})p(x_s \mid z^\prime_{s})p(x_{(s+1):t} \mid z^\prime_s) },$$
;;
;; which are costly to compute. This cost arose due to the fact that we must
;; compute the marginal probability of the future of our observations sequence
;; starting from our current state $z_s$, that is $p(x_{(s+1):t} \mid z_s)$.
;; Doing this involves a sum over all possible state sequences over the suffix
;; of our observation sequence.
;;
;; What can we do about this? One idea is to use a different **local**
;; distributions that don't need to know about the future:
;;
;; $$p(z_s \mid z_{s-1}, x_{s}) = \frac{p(z_s \mid z_{s-1})p(x_s \mid z_{s})}
;;         { \sum_{z^\prime_s} p(z^\prime_s \mid z_{s-1})p(x_s \mid z^\prime_{s})}.$$
;;
;; These distributions are locally optimal in some sense since they make optimal
;; use of all information up to time $s$, but don't need to know the future.
;; Using such distributions, we could define the probability of our entire,
;; joint state sequence as,
;;
;; $$q(z_{1:t} | x_{1:t}) = p(z_1 \mid x_1) \prod_{s=2}^t p(z_s \mid z_{s-1}, x_{s}).$$
;;
;; Note that we did not write $p(z_{1:t} | x_{1:t})$, in general $q(z_{1:t} |
;; x_{1:t}) \neq p(z_{1:t} | x_{1:t})$, this product of distributions does not
;; recover our true posterior, even though $p(z_s \mid z_{s-1}, x_{s}) =
;; \sum_{x_{(s+1):t}} p(z_s, x_{(s+1):t} \mid z_{s-1}, x_{s})$ is a correct
;; maginal with respect to our joint distribution $p$.
;;
;; Still, it seems like it might be a reasonable thing to do&mdash;it makes the
;; best use possible of all the information so far, and in many domains where we
;; want to use sequential sampling techniques, the data themselves arrive one at
;; a time and thus, we *cannot* know the future. In that case, this is indeed
;; the best we could do.
;;
;; Is there a way to make use of such a distribution in a way that that
;; ultimately will draw correct samples from $p(z_{1:t} | x_{1:t})$?
;;

;; # Sequential Importance Sampling
;;
;; In the last sections, we discussed how samples from a posterior distribution
;; could be sequentialized using the chain rule, allowing us to draw joint
;; samples from a sequence of conditional distribution $p(z_s \mid z_{s-1},
;; x_{s:t})$. We also noted that computing this sequence of distributions is
;; costly, since it inolves summing out all future paths over states that
;; generate the suffix of our observation sequence. We introduced the notion of
;; a *locally optimal* sequence of distributions, which makes optimal use of
;; information, ignoring the future.
;;
;; How can we make the use of such a locally optimal distribution?
;;
;; Of course, we adopt the approach we have throughout the course: We assume
;; some proposal distribution $q$ and we correct it to the true distribution. In
;; the case of particle filtering, we adopt a sequentialized version of
;; importance sampling.
;;
;; Suppose we wished to draw importance samples from our target posterior
;; $p(z_{1:t}, x_{1:t})$, from some proposal distribution $q(z_{1:t} \mid
;; x_{1:t})$ if we did this each sample would be weighted using the following
;; weight by standard importance sampling:
;;
;; $$w_{1:t} = \frac{p(z_{1:t}, x_{1:t})}{q(z_{1:t} \mid x_{1:t})}$$
;;
;; Let's further assume (beyond standard importance sampling) that our proposal
;; is *incremental*, that is we have an initial proposal distribution $q(z_1
;; \mid x_1)$ and a sequence of proposals $q(z_t \mid z_{t-1}, x_t)$ w such that
;;
;; $$q(z_{1:t} \mid x_{1:t}) = q(z_1 \mid x_1) \prod_{s=2}^t q(z_s \mid z_{s-1}, x_s)$$
;;
;; With this sequentialized proposal distribution, our weight calculations can
;; also be sequentialized.
;;
;; $$\begin{align}
;; w_{1:t} &= \frac{p(z_{1:t}, x_{1:t})}{q(z_{1:t} \mid x_{1:t})}\\
;; w_{1:t} &= \frac{p(z_{1:{t-1}}, x_{1:t-1})p(z_t \mid z_{t-1})p(x_t \mid z_t)}{q(z_{1:t} \mid x_{1:t})}\\
;; w_{1:t} &= \frac{p(z_{1:{t-1}}, x_{1:t-1})p(z_t \mid z_{t-1})p(x_t \mid z_t)}{q(z_{1:t-1} \mid x_{1:t-1})q(z_t \mid z_{t-1}, x_t)}\\
;; w_{1:t} &= \frac{p(z_{1:{t-1}}, x_{1:t-1})}{q(z_{1:t-1} \mid x_{1:t-1})}\frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{q(z_t \mid z_{t-1}, x_t)}\\
;; w_{1:t} &= w_{1:t-1}\frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{q(z_t \mid z_{t-1}, x_t)}\\
;; \end{align}
;; $$
;;
;; We can now keep an updated set of hypotheses at each time step, called
;; *particles* by sequentially sampling extensions to the hypotheses at the last
;; time step using $q(z_t \mid z_{t-1}, x_t)$ and udating the weight of each
;; particle by multiplying by
;;
;; $$\frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{q(z_t \mid z_{t-1}, x_t)}$$.
;;
;; Sequential important sampling gives us a way to use an incremental proposal
;; distribution to update a hypothesis as we see more data points. It forms the
;; core of the algorithm we conser next: *particle filtering*.

;; # Particle Filters
;;
;; *Particle filters* are a widely-used variant of SMC where we represent an
;; *approximation to our posterior distribution $p(z_{1:1} | x_{1:t})$ as a set
;; *of samples of trajectories up to that time step, where each sampled
;; *trajectory is called a *particle*.
;;
;; In the setting of Gen, a particle just corresponds to a trace containing a
;; sampled state up to time step $t$.
;;
;; We implement a particle filter using the functions
;; `initialize_particle_filter` and `particle_filter_step!` (note that this
;; implementation uses the *standard proposal distribution*, which we discuss in
;; more detail below).
;;

;; ```julia
;; init_obs = Gen.choicemap((:x => 1, ground_truth[:x => 1]))

;; n_particles = 20
;; state = Gen.initialize_particle_filter(
;;     hmm,
;;     (1, ϕ, transition, observation),
;;     init_obs,
;;     n_particles)

;; ;;=
;; (log_incremental_weights,) = particle_filter_step!(
;;     state::ParticleFilterState,
;;     new_args::Tuple,
;;     argdiffs,
;;     observations::ChoiceMap,
;;     proposal::GenerativeFunction,
;;     proposal_args::Tuple)
;; =;;

;; for t=2:T
;;     obs = Gen.choicemap((:x => t, ground_truth[:x => t]))
;;     Gen.particle_filter_step!(
;;         state,
;;         (t, ϕ, transition, observation),
;;         (UnknownChange(),),
;;         obs)
;; end;
;; state
;; ```
;; State objects contain three pieces of information:
;;
;;  1. A vector of traces for each particle at the current time step.
;;  2. A vector of weights for each particle at the current time step.
;;  3. An average (log) weight (more on this later).
;;
;; Let's implement a function which runs our particle filter.
;;

;; ```julia
;; function hmm_particle_filter(
;;         ground_truth::Gen.DynamicDSLTrace,
;;         n_particles::Int64;
;;         resampling=false)

;;     init_obs = Gen.choicemap((:x => 1, ground_truth[:x => 1]))

;;     state = Gen.initialize_particle_filter(
;;         hmm,
;;         (1, ϕ, transition, observation),
;;         init_obs,
;;         n_particles)

;;     all_particle_weights=reshape(ones(n_particles),(1,:))
;;     all_lmls=[]

;;     for t=2:T
;;         if resampling
;;             Gen.maybe_resample!(state, ess_threshold=n_particles/2)
;;         end

;;         obs = Gen.choicemap((:x => t, ground_truth[:x => t]))

;;         Gen.particle_filter_step!(
;;             state,
;;             (t, ϕ, transition, observation),
;;             (UnknownChange(),),
;;             obs)

;;         all_particle_weights=vcat(
;;             copy(all_particle_weights),
;;             reshape(copy(state.log_weights), (1,:)))

;;         push!(all_lmls, copy(state.log_ml_est))
;;     end

;;     return(state, all_particle_weights, all_lmls)
;; end;
;; ```

;; It will be useful to plot the outputs of our particle filters.

;; ```julia
;; function logsumexp(a; dims=:)
;;     m = maximum(a, dims=dims) # pull out max for stability
;;     m .+ log.(sum(exp.(a .- m), dims=dims))
;; end;
;; normalize_weights(W) = W .- logsumexp(W; dims=2);

;; function plot_particle_filter(
;;         T, state, all_particle_weights,
;;         all_lmls;
;;         backend=plotly())

;;     backend;
;;     n_particles=length(state.log_weights)
;;     normalized_weights = normalize_weights(all_particle_weights[1:T,:]);
;;     labels=reshape(["Particle $j" for j in 1:n_particles],(1,:))
;;     observation_markerstyle=(:circle, 4, 0.7, stroke(1, .5, :gray))
;;     state_markerstyle=(:square, 4, 0.7, stroke(1, .5, :gray))

;;     # Get observations
;;     observations = [[get_choices(state.traces[j])[:x=>i] for i in 1:T] for j in 1:n_particles]
;;     observations_plot = scatter(observations, title="Observations", ylabel="value", #xlabel="t",
;;         yminorgrid=true, yminorticks=2, ygridalpha=.3, minorgridalpha=.2,
;;         marker=observation_markerstyle, alpha=.4)

;;     # Get all the particles paths through the trellis.
;;     paths = [[get_choices(state.traces[j])[:z=>i] for i in 1:T] for j in 1:n_particles]
;;     paths_plot = plot(paths, title="Particles' paths", ylabel="hidden state", #xlabel="t",
;;         yminorgrid=true, yminorticks=2, #ygridalpha=.3, minorgridalpha=.2,
;;         marker=state_markerstyle, lw=3, alpha=.4)

;;     # And weights
;;     unnormalized_weights_plot = plot(all_particle_weights[1:T,:],
;;         title="Particles' weights (at time t)", ylabel="log weight", #xlabel="t",
;;         ticks=:native, lw=4, alpha=.5)

;;     weights_plot_log = plot(normalized_weights[1:T,:],
;;         title="Particles' log normalized weights (at time t)", ylabel="log weight", #xlabel="t",
;;         ticks=:native, lw=4, alpha=.5)

;;     weights_plot = plot(exp.(normalized_weights)[1:T,:],
;;         title="Particles' normalized weights (at time t)", ylabel="weight", xlabel="t",
;;         ticks=:native, lw=4, alpha=.5)

;;     plot(observations_plot,paths_plot,unnormalized_weights_plot,weights_plot,weights_plot_log,
;;         xlims=(0,T+1), layout=(5,1), size=(1000,600), legend = false, label=labels)
;; end;

;; function make_plot_animation(
;;         T,
;;         ground_truth,
;;         n_particles;
;;         initialization=nothing,
;;         proposal=nothing,
;;         resampling=true,
;;         rejuvenation=true,
;;         rejuv_proposal=nothing,
;;         rejuv_proposal_args=nothing,
;;         fps = 10)

;;     init_obs = Gen.choicemap((:x => 1, ground_truth[:x => 1]))

;;     state = if initialization==nothing
;;       Gen.initialize_particle_filter(
;;         hmm,
;;         (1, ϕ, transition, observation),
;;         init_obs,
;;         n_particles)
;;     else
;;         Gen.initialize_particle_filter(
;;             hmm,
;;             (1, ϕ, transition, observation),
;;             init_obs,
;;             initialization,
;;             (ϕ, observation, ground_truth[:x => 1],),
;;             n_particles)
;;     end

;;     all_particle_weights=reshape(ones(n_particles),(1,:))
;;     all_lmls=[]

;;     anim = @animate for t=2:T


;;         if resampling
;;             Gen.maybe_resample!(state, ess_threshold=n_particles/2)
;;         end

;;         obs = Gen.choicemap(
;;             (:x => t, ground_truth[:x => t]))

;;         if proposal==nothing
;;             Gen.particle_filter_step!(
;;                 state,
;;                 (t, ϕ, transition, observation),
;;                 (UnknownChange(),),
;;                 obs)
;;         else
;;             Gen.particle_filter_step!(
;;                 state,
;;                 (t, ϕ, transition, observation),
;;                 (UnknownChange(),),
;;                 obs,
;;                 proposal,
;;                 (ground_truth[:x => t],))
;;         end

;;         if rejuvenation
;;             for i=1:n_particles
;;                 choices = select([:z => i for i in 1:T]...)
;;                 if rejuv_proposal==nothing
;;                     state.traces[i], _  = mh(state.traces[i], choices)
;;                 else
;;                     state.traces[i], _  = mh(state.traces[i], rejuv_proposal, rejuv_proposal_args)
;;                 end
;;             end
;;         end

;;         all_particle_weights=vcat(
;;             copy(all_particle_weights),
;;             reshape(copy(state.log_weights), (1,:)))

;;         push!(all_lmls, copy(state.log_ml_est))

;;         plot_particle_filter(
;;             t, state, all_particle_weights, all_lmls;
;;             backend=gr())
;;     end

;;     gif(anim, fps = fps)
;; end;
;; ```

;; We will condition on the observations contained in our sampled *ground truth*
;; trace.

;; ```julia
;; get_choices(ground_truth)

;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10,
;;     resampling=false,
;;     rejuvenation=false,
;;     fps=6)
;; ```

;; What are the bottom three plots in the figure? Our discussion of particle
;; filtering so far has left out an important aspect of the approach: the use of
;; arbitrary proposal distributions. Before discussing the behavior of these
;; algorithms more generally, we introduce the proposal distribution we are
;; using.

;; First, we did not specify above, but our implementation of the HMM particle
;; filter here is using the *standard proposal* which is just the prior
;; distribution, $q(z_t \mid z_{t-1}, x_{t})=p(z_t \mid z_{t-1})$.
;;
;; <!-- , rather than the locally optimal proposal $q(z_t \mid z_{t-1}, x_{t})=\frac{p(z_t \mid z_{(t-1)})p(x_t \mid z_{t})}
;;         { \sum_{z_t} p(z_t \mid z_{(t-1)})p(x_t \mid z_{t})}$.-->
;;
;; Similar to the way that the `generate` function worked, this gives rise to an
;; incremenetal form of *likelihood weighting*. Our incremental weights are
;; given by
;;
;; $$\begin{align}
;; w_t &= \frac{p(z_t\mid z_{t-1})p(x_t \mid z_t)}{q(z_t \mid z_{t-1}, x_{t})}\\
;; w_t &= \frac{p(z_t\mid z_{t-1})p(x_t \mid z_t)}{p(z_t\mid z_{t-1})}\\
;; w_t &= p(x_t \mid z_t)
;; \end{align}$$
;;
;; Thus the total weight up to time point $t$ is just the likelihood of the
;; observations under the path of the states.
;;
;; $$w_{1:t} =  \prod_{s=1}^{t} p(x_{s} \mid z_s)$$
;;
;; Let's look at the behavior of the weights for this incremental
;; likelihood-weighted HMM particle filter.
;;
;; First, examining the third panel, we notice that the weight of each particle
;; is going down over time as the overall likelihood of the sequence goes down.
;;
;;
;; > Why is this the expected behavior?
;;
;; It is also useful to ask how the particles behave with respect to one
;; another, rather than globally. We can examine this question by looking at the
;; normalized particle weights at each time step. The fourth panel shows the
;; normalized weights, which allows us to examine the relative quality (in this
;; incremental likelihood) of each sampled state trajectory. The fith panel
;; shows the log normalized weights, which allows us to see more clearly the
;; spread of particle weights over time.
;;
;; We see in the fourth plot, that the quality of particles is very peaky. In
;; particular, particles are exhibiting a switching behavior with one particle
;; at a time dominating the others in terms of its quality. Why is this?
;;
;; Since the proposal distribution pays no attention to the likelihood of each
;; observation, the particles transition based just on the prior probability of
;; the transition. Our transition distributions are relatively flat, while our
;; observation distributions are sparse.

;; ```julia
;; round.(transition; digits=3)

;; round.(observation; digits=3)
;; ```

;; Thus, it is only a matter of time before a some particle transitions to a bad
;; state that generates the observed data with low probability. There is nothing
;; to keep the particles transitioning to good states and thus their weights
;; follow a random walk, for a while, they will accidentally transition to good
;; states, and then randomly transition to bad states. Since the observation
;; distribution is so peaky, typically one particle will dominate the rest.
;;
;; How can we fix this behavior?
;;
;; As a first attempt, we could use a more sensible proposal distribution. Which
;; one could we use? One obvious answer to make use of the likelihood of each
;; observed symbol in our proposals. We have already seen a proposal which does
;; this the *locally optimal proposal*.
;;
;; Consider the sequence of distributions:
;;
;; $$p(z_t \mid z_{1:(t-1)}, x_{1:t}) = \frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{\sum_{z_{t}} p(z_t \mid z_{t-1})p(x_t \mid z_t)} = p(z_t \mid z_{t-1}, x_{t})$$
;;
;; Note that the normalizing constant $\sum_{z_{t}} p(z_t \mid z_{t-1})p(x_t
;; \mid z_t)$ is the conditional marginal probability of the observation,
;; conditioned on the last state, that is, $p(x_t \mid z_{t-1})$.
;;
;; Using $p(z_t \mid z_{t-1}, x_t)$ as $q(z_t \mid z_{t-1}, x_t)$, the locally
;; optimal proposal is optimal with respect to time step $t$, that is, assuming
;; we can't see into the future.
;;
;; Plugging this into our formula for weights we get
;;
;; $$w_{t} = \frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{\frac{p(z_t \mid z_{t-1})p(x_t \mid z_t)}{\sum_{z_{t}} p(z_t \mid z_{t-1})p(x_t \mid z_t)}} = \sum_{z_{t}} p(z_t \mid z_{t-1})p(x_t \mid z_t) = p(x_t \mid z_{t-1})$$
;;
;; In other words, the incremental weight for the optimal proposal is just the
;; marginal probability of the observation given the last state. Thus, so long
;; as the preceding state can assign high probability to the next observation
;; **using some state** our weights will be high.
;;
;; $$w_{1:t} =  \prod_{s=1}^{t} p(x_{s} \mid z_{s-1})$$
;;
;; To implement this, we use versions of
;; `initialize_particle_filter` and `particle_filter_step!` which take generative functions implementing the initial and incremental proposals.

;; ```julia
;; function normalize(a)
;;     a ./ sum(a)
;; end;

;; @gen function optimal_proposal_init(ϕ, observation, next_obs::Int)
;;     ws = Vector{Float64}(undef, length(ϕ))
;;     for (i, val) in enumerate(ϕ)
;;         ws[i]=ϕ[i]*observation[i,next_obs]
;;     end

;;     weights=normalize(ws)
;;     ({:z => 1} ~ categorical(weights))
;; end

;; @gen function optimal_proposal(old_particle::Gen.DynamicDSLTrace, next_obs::Int)
;;     T, ϕ, transition, observation = get_args(old_particle)

;;     transitions=transition[old_particle[:z => T],:]

;;     ws = Vector{Float64}(undef, length(transitions))
;;     for (i, val) in enumerate(transitions)
;;         ws[i]=transitions[i]*observation[i,next_obs]
;;     end

;;     weights=normalize(ws)
;;     ({:z => T+1} ~ categorical(weights))
;; end;
;; ```

;; Let's look at the behavior of this particle filter over time.


;; ```julia
;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10,
;;     initialization=optimal_proposal_init,
;;     proposal=optimal_proposal,
;;     resampling=false,
;;     rejuvenation=false,
;;     fps=6)
;; ```

;; We can see now that all four particles are staying relatively probable
;; throughout. This is because our transition distribution is relatively flat
;; which allows our locally optimal proposal to choose transitions to good
;; states that assign high probability to the next observation What would happen
;; though if our transition distribution were also sparse.

;; ```julia
;; α = fill(0.01,K)

;; transition = zeros(K,K)
;; for k in 1:K
;;     transition[k,:] = rand(Dirichlet(α))
;; end;
;; ```

;; ```julia
;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10,
;;     initialization=optimal_proposal_init,
;;     proposal=optimal_proposal,
;;     resampling=false,
;;     rejuvenation=false,
;;     fps=6)
;; ```

;; We see a return to our original bad behavior. The optimal proposal cannot
;; make good transitions from bad preceding states in this case, and our
;; particles get "stuck" in poor trajectories. How can we design an algorithm to
;; do better than this.

;; # The Bootstrap Filter
;;
;; One approach to eliminating bad trajectories is the *bootstrap fiter*. With
;; the bootstrap filter we simply add an importance resampling step whenever the
;; set of particles becomes too skewed. One way of doing this is to use the
;; *effective sample size* ESS.
;;
;; $$\mathrm{ESS} = \frac{1}{\sum_{p} \hat{w}_p^2}$$
;;
;; Where $p$ ranges over particles and $\hat{w}_p$ is the normalized importance
;; weight of particle $p$. A typical criterion is to insert an importance
;; resampling step each time the ESS drops below the number of particles divided
;; by $2$.
;;
;; Gen provides the `maybe_resample!` function to implement this bootstrapping step.

;; ```julia
;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10,
;;     initialization=optimal_proposal_init,
;;     proposal=optimal_proposal,
;;     resampling=true,
;;     rejuvenation=false,
;;     fps=6)
;; ```

;; We can see here that now the particles are staying almost perfectly
;; equiprobable thanks to the resampling steps.

;; # Rejuvenation
;;
;; The bootstrap filter improved the performance of our particle filter by
;; ensuring that extremely bad particles were "weeded out" and replaced by
;; particles which had a better trace. However, we now see an additional
;; problem&mdash;there is very little diversity in the hypothesized latent
;; variables in our particle states.

;; ```julia
;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10,
;;     initialization=optimal_proposal_init,
;;     proposal=optimal_proposal,
;;     resampling=true,
;;     rejuvenation=true,
;;     fps=6)

;; @gen function gibbs_proposal(trace, addr_prefix)
;;     T, ϕ, transition, observation = get_args(trace)
;;     K=size(transition)[1]

;;     # Update each latent variable in turn
;;     for i=1:T

;;         address = addr_prefix => i

;;         lps = Vector{Float64}(undef, K)
;;         for j in 1:K
;;             obs = Gen.choicemap()
;;             obs[address] = j
;;             (t, _) = Gen.update(trace, obs)
;;             lps[j]=get_score(t)

;;         end

;;         m = maximum(lps)
;;         lse = m .+ log.(sum(exp.(lps .- m)))
;;         probs=exp.(lps .- lse)
;;         ps = normalize(round.(probs, digits = 3))

;;         {address} ~ categorical(ps)
;;     end
;; end


;; make_plot_animation(
;;     T,
;;     ground_truth,
;;     10, #num particles
;;     initialization=optimal_proposal_init,
;;     proposal=optimal_proposal,
;;     resampling=true,
;;     rejuvenation=true,
;;     rejuv_proposal=gibbs_proposal,
;;     rejuv_proposal_args=(:z,),
;;     fps=6)
;; ```
