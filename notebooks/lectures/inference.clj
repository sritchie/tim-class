;; # Inference
;;
;; We have introduced generative functions in Julia, as well as related them to
;; probability theory. So far, however, we have only looked at sampling in the
;; forward direction&mdash;from the joint. However, the fundamental reason that
;; these tools are useful is that they can be used in the *reverse*
;; direction&mdash;making inferences about unseen parts of our model based on
;; observed information. In simulation-based approaches to probabilistic
;; modeling, this is accomplished via *conditioning* our models on the values of
;; particular outcomes of random variables. Most of the rest of the course will
;; be devoted to understanding how to efficiently approximate conditional
;; probability distriibutions.
;;
;; In a Bayesian setting we use the term *inference* to refer to a kind of
;; hypothetical reason&mdash;implemented using conditional probabilities.
;; Typically, this kind of reasoning asks for likely values of some unobserved
;; random variables in the model given some particular values of some *observed*
;; random variables.
;;
;; Let's begin by introducing a few example inference problems to work with.

;; ```julia
;; using Gen
;; using Plots
;; ```

;; # Latent and Observed Random Variables
;;
;; Consider this variant of our biased coin model.
;;

;; ```julia
;; @gen function flip_biased_coin(N)
;;     θ  ~ beta(1,1)

;;     [{:x => i} ~ bernoulli(θ)  for i in 1:N]
;; end;

;; t=Gen.simulate(flip_biased_coin,(5,))
;; Gen.get_choices(t)
;; ```

;; When we are using probabilistic models, we make a distinction between
;; *latent* and *observed* random variables. The observed random variables are
;; those variables which represent place holders for datapoints in our model. In
;; the biased coin model, we generate $N$ observed coin flips. The random
;; variables $X_i$ representing these $N$ coin flips constitute our observed
;; random variables. It is important to be careful here: The observed RV is the
;; placeholder, not a particular observed value that we see in some specific
;; dataset. *Latent* random variables (also called *hidden random variables*)
;; are those random variables who value must have been sampled in order to
;; generate our data, but whose value we don't think of as available in our
;; data. In the biased coin model, the single latent random variable is
;; $\theta$.
;;
;; Note that which variables in a model are considered as observed and latent is
;; a matter of application and can change for different datasets for the same
;; model. The observed variables are just those variables for which you have
;; values available in that particular application.
;;
;; > What's an example of a scenario where for one application an RV might be observed and for another it might be latent?
;;
;; The distinction between observed and latent random variables allows us to
;; formulate a precise, general meaning of the word *inference* in a Bayesian
;; setting: Inference is reasoning about possible values for the latent random
;; variables from particular values for the observed random variables.

;; # Inference Problems
;;
;; In our biased coin flipping model, we return just the values of the $N$ coin
;; flips. However, $\theta$ is unobserved. Thus a natural inference problem is
;;
;; > What is the value of $\theta$ given the $N$ coin tosses that we have seen?
;;
;; In a Bayesian setting, we typically try to quantify our uncertainty over our
;; guesses, so rather than asking for a specific value of $\theta$ we can ask
;; for a distribution over values of $\theta$.
;;
;; > What is the distribution over $\theta$ given the $N$ coin tosses that we have seen?
;;
;; To understand this better, consider the following question.
;;
;; > If the distribution over $\theta$ was represented as a beta distribution
;; > like those below, what would it look like if $29/100$ of the outcomes were
;; > heads? What if $79/100$ of the outcomes were heads?
;;
;;
;; <img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg"
;;      alt="Beta Distribution"
;;      width="500" height="500"/>
;;
;; These last two questions are standard examples of the problem of *posterior
;; inference* in probabilistic modeling. The term *posterior* refers to the
;; distribution of a latent RV **after** observing some data points. We would
;; like to compute the *posterior distribution* over latent random variables
;; given the observed random variables.
;;
;; There is a second kind of inference problem which is commonly considered in Bayesian statistics.
;;
;; > What is the distribution over the next coin flip, given the preceding coin flips.
;;
;; This is the statement of another a type of inference problem often called
;; *posterior predictive inference*. The *posterior predictive distribution* is
;; the distribution over next observations given the preceding observations,
;; averaging over posterior values of the latent random variables. In order to
;; model the posterior predictive distribution, we will have to introduce a new
;; latent random variable $X_{N+1}$, as a placeholder for the next coin flip
;; after observing the previous coin flips.
;;
;; # Inference via Conditioning
;;
;; We have seen an inference problem stated as finding the distribution of some
;; unknown random variables given some known random variables. But what **is**
;; the distribution over these unknown random variables? In each case, we are
;; looking for a *conditional distribution* over the unknown variables given the
;; values of the known variables.
;;
;; In the simulation unit, we talked about the interpretation of conditional
;; probabilities.
;;
;; >  For a conditional distribution, we are **required** to use the value of
;; >  the random variable specified in the condition; we absolutely **must
;; >  have** this information to draw our sample.
;;
;; In Bayesian statistics, we represent posterior and posterior inference
;; problems using conditional distributions. For instance, when the distribution
;; of $\theta$ given some observed coin tosses is represented by
;;
;; $$\Pr(\theta \mid \{X_{i}\}_{i=1}^N, \alpha, \beta ).$$
;;
;; Clearly if we are reasoning about plausible values of $\theta$ given our
;; data, then the $\{X_{i}\}_{i=1}^N$ is required.
;;
;; However, we now have a problem. The model definition which was given by the
;; following joint expression
;;
;; $$\Pr(\theta, \{X^{(i)}\}_{i=1}^N \mid \alpha, \beta) = \Pr(\theta \mid \alpha, \beta) \prod_{i=1}^N \Pr(X^{(i)}\mid \theta).$$
;;
;; All the conditional distributions here, $\Pr(X_{i}\mid \theta)$ are simply
;; part of the definition of the model.
;;
;; However, the conditional distribution above is not part of the model
;; definition, For unknown conditional distributions such as this, we must
;; compute their values using the laws of probability theory. In the next
;; sections we will see how to do this.

;; # Operations of Probability Theory
;;
;; We have set ourselves the goal of computing posterior distributions such as
;;
;; $$\Pr(\theta \mid \{X_{i}\}_{i=1}^N, \alpha, \beta )$$
;;
;; Given a particular generative model, we always start with a joint
;; distribution over random variables such as
;;
;; $$\Pr(\theta \mid \alpha, \beta) \prod_{i=1}^N \Pr(X_{i}\mid \theta)$$
;;
;; How do we compute the first distribution starting from the second
;; distribution as input. Probability theory makes use of two operations which
;; we can use to do this: *marginalization* and *conditioning*. Since
;; conditioning is defined in terms of marginalization, we will start with the
;; former.

;; # Marginalization
;;
;; For a given joint distribution, say $\Pr(A,B)$, a *marginal distribution* is
;; some distribution that only contains a subset of the random variables in the
;; joint, such as $\Pr(A)$ and $\Pr(B)$.
;;
;; More generally, for some set of random variables $\mathbf{X}=\{X_{1}, \dots,
;; X_{K}\}$ there are $|\mathcal{P}(\mathbf{X})|-1 = 2 ^{K}-1$ possible marginal
;; distributions.
;;
;; How are marginal distributions related to joint distributions?
;;
;; The definition of marginalization is given by
;;
;; $$\Pr(A=a) = \sum_{b\in B} \Pr(A=a, B=b)$$
;;
;; if $B$ is a discrete random variable or if $B$ is a continuous random
;; variable
;;
;; $$\Pr(A) = \int \Pr(A, B=b) db$$
;;
;; More generally
;;
;; $$\Pr(X_{1}, \dots, X_{M})= \sum_{X_{M+1}} \dots \sum_{X_{K}} \Pr(X_{1}, \dots, X_{M}, X_{M+1}, \dots, X_{K})$$
;;
;; where the sums are replaced by integrals where appropriate. Often the following notation is used.
;;
;; $$\Pr(X_{1}, \dots, X_{M})= \sum_{X_{M+1},\dots,X_{K}} \Pr(X_{1}, \dots, X_{M}, X_{M+1}, \dots, X_{K})$$
;;
;; In other words, marginalization refers to the process of summing over all of
;; the values of some of the random variables in a joint distribution.
;;
;; Marginal distributions are called as such because if we think about the tabular view of joint distributions, they can be written in the "margins" of the table. Returning to our `flip_two` example from the last unit.
;;
;; |    $$X_{1}\backslash X_{2}$$  |`true`|`false`| $$\Pr(X_{1})$$|
;; |---      |---|---|---|
;; | `true`  |  $$p(\texttt{true},\texttt{true})$$ |$$p(\texttt{true}, \texttt{false})$$| $$p_{X^{(1)}}(\texttt{true})$$|
;; | `false` |  $$p(\texttt{false}, \texttt{true})$$ |$$p(\texttt{false},\texttt{false})$$| $$p_{X^{(1)}}(\texttt{false})$$|
;; | $$\Pr(X_{2})$$ |  $$p_{X_{2}}(\texttt{true})$$ |$$p_{X_{2}}(\texttt{false})$$| |
;;
;; We say that we marginalize *over* a variable like $X_{1}$ or we *marginalize* $X_{1}$ *away*.
;;
;; Intuitively, **marginalization is a way of getting rid of random variables in a joint distribution**.
;;

;; # Samplers, Scorers, and Marginalization
;;
;; In the last unit, we introduced the idea that there were two different kinds representations for probability distributions: *scorers* which were functions that returned the density of points in the sample space of the distribution and *samplers* which are function which return elements from the sample space of the distribution at a rate proportional to their densities.
;;
;; In the definitions of marginalization above, we implicitly took the scoring view: we defined a procedure for taking a scorer for a joint distribution and returning a scorer for the target marginal. Thus, we can think of marginalization as a higher order operation that takes a scorer representation of a joint distribution, as well as a specification of which variables to preserve and returns a scorer representation of the desired marginal:
;;
;; $$\texttt{scorer} \times\ \{X_{1},\dots,X_{M}\} \rightarrow \texttt{scorer}$$
;;
;; Scorer marginalization can be a very expensive operation as it requires us to do a sum over every value of every random variable that we wish to marginalize away. If these random variables have large supports and there are many of them, then we will be marginalizing over a cartesian product which can be extraordinarily large.
;;
;; Viewing marginalization as a higher order function from scorers to scorers raises an interesting quesiton:
;;
;; > What is marginalization viewed as a higher order function from samplers to samplers?
;;
;; It turns out that marginalization for samplers is much easier than in the case of scorers.
;; If we have a generative function that samples from some joint distribution marginalization corresponds to simply forgetting, ignoring, or not returning the values of the  random variables that have been marginalized away.
;;
;; In other words, we can simply draw a sample from our joint and only return the desired target variables in $\{X_{1},\dots,X_{M}\}$.
;;
;; $$\texttt{sampler} \times\ \{X_{1},\dots,X_{M}\} \rightarrow \texttt{sampler}$$
;;
;; One important question we need to ask is if this gives a *correct sampler*. A sampler is correct if it returns values at rates proportional to their densities.
;;
;; > Does forgetting give us a correct sampler for  marginal distributions?
;;
;; We see here our first suggestion of why the sampling perspective may be  computationally adventageous in some cases. While scoring a marginal probability over a set of random variables can be exponentially costly, sampling from the same marginal is typically quite cheap.

;; # Conditioning
;;
;; We have discussed how joint distributions such as $\Pr(A,B)$ are related to marginal
;; distributions such as $\Pr(A)$ and $\Pr(B)$, and the operation
;; that relates them: marginalization. We are now in a position to
;;  introduce another
;; fundamental operation from probability theory: *conditionalization* or *conditioning*.  Conditioning as an operation can be thought of as a
;; form of hypothetical reasoning where we ask about the probability
;; distribution over some (set of) random variables, assuming some other
;; random variables take on some particular values. For instance, we
;; might ask what the probability distribution is over $X_{2}$ given
;; that $X_{1}=\texttt{true}$. We write the conditional probability as $\Pr(X_{2}=x_{2}
;; \mid X_{1}=\texttt{true})$. The expression to the right of the conditioning bar is
;; called the *conditioner* and can be any predicate that we require to
;; be true.
;;
;; For discrete random variables, conditioning is defined as follows:
;;
;; $$\Pr(X=x | Y=y) = \frac{\Pr(X=x,Y=y)}{\Pr(Y=y)} = \frac{\Pr(X=x,Y=y)}{\sum_{x^{\prime} \in X} \Pr(X=x^{\prime},Y=y)}$$
;;
;; or more generally
;;
;;
;; $$\Pr(X_{1}, \dots, X_{M} \mid X_{M+1}=x_{M+1}, \dots, X_{K}=x_{K})= \frac{Pr(X_{1}, \dots, X_{M}, X_{M+1}, \dots, X_{K})}{\sum_{X_{1}} \dots \sum_{X_{M}} \Pr(X_{1}, \dots, X_{M}, X_{M+1}, \dots, X_{K})}=\frac{Pr(X_{1}, \dots, X_{M}, X_{M+1}, \dots, X_{K})}{\Pr(X_{M+1}, \dots, X_{K})}$$
;;
;;
;; We can think of conditioning as a two step process. First, we get the
;; subset of joint states where the conditioner is true, that is, removing all of the states where the conditioner is
;; false. We then renormalize the joint probabilities of these states so
;; that they are a probability distribution again by dividing through by
;; the marginal probability of the conditioner.
;;
;;
;;
;;
;; One can think of
;; conditioning as "zooming in" on the part of the joint space we are
;; interested in (the part where the conditioner is true) and then making
;; the result a probability distribution by renormalizing in order to make things add
;; to $1$.
;;
;;
;; Let's consider the following joint distribution again.
;;
;; |    $$X_{1}\backslash X_{2}$$  |`true`|`false`| $$\Pr(X_{1})$$|
;; |---      |---|---|---|
;; | `true`  |  $$p(\texttt{true},\texttt{false})$$ |$$p(\texttt{true}, \texttt{false})$$| $$p_{X_{1}}(\texttt{true})$$|
;; | `false` |  $$p(\texttt{true}, \texttt{false})$$ |$$p(\texttt{false},\texttt{false})$$| $$p_{X_{1}}(\texttt{false})$$|
;; | $$\Pr(X_{2})$$ |  $$p_{X_{2}}(\texttt{true})$$ |$$p_{X_{2}}(\texttt{false})$$| |
;;
;; Suppose that we want to ask about the conditional distribution $\Pr(X_{1} \mid X_{2}=\mathtt{false})$. In this case, we only care about the final column of the joint distribution.
;;
;;
;; |    $$X_{1}\backslash X_{2}$$  |`true`|`false`|
;; |---      |---|---|
;; | `true`  |  - |$$p(\texttt{true}, \texttt{false})$$|
;; | `false` |  - |$$p(\texttt{false},\texttt{false})$$| $$p(\texttt{false})$$|
;; | $$\Pr(X_{2})$$ |  - |$$p_{X_{2}}(\texttt{false})$$|
;;
;;
;; To make this into a conditional distribution we renormalize.
;;
;; |    $$X^{(1)}\backslash X^{(2)}$$  |`true`|`false`|
;; |---      |---|---|
;; | `true`  |  - |$$\frac{p(\texttt{true}, \texttt{false})}{p_{X_{2}}(\texttt{false})}$$|
;; | `false` |  - |$$\frac{p(\texttt{false},\texttt{false})}{p_{X_{2}}(\texttt{false})}$$| $$p(\texttt{false})$$|
;;
;; Notice what this means: The conditional probability of an outcome is related to the joint probability of the same outcome by a normalizing constant&mdash;since the probability of the conditioner is the same for all values of the conditioned variables. That is, whatever states remain in the distribution after condititioning have the same probabilities proportional to one another, even though their absolute probabilities may change. Conditioning is&mdash;in a sense that can be made precise&mdash;the maximally *conservative* operation one can perform to implement hypothetical reasoning and still obey the laws of probability theory. Nevertheless, despite its simplicity, conditioning can have surprisingly rich consquences.
;;
;; The fact that the joint and conditional are related by a normalizing constant will prove to be useful in defining a number of algorithms we will explore in this course.
;;

;; # Samplers, Scorers, and Conditioning
;;
;; Above, we defined conditioning once again in terms of scoring. It can be thought of as a higher order function which takes a scorer representation of a distribution as well as the random variables and values to condition on
;;
;; $$\texttt{scorer} \times\ \{X_{M+1}=x_{M+1}, \dots, X_{K}=x_{K} \} \rightarrow \texttt{scorer}$$
;;
;; > What is the sampling equivalent to conditioning?
;;
;; $$\texttt{sampler} \times\ \{X_{M+1}=x_{M+1}, \dots, X_{K}=x_{K} \} \rightarrow \texttt{sampler}$$
;;
;; As we will see below, this does not have nearly as simple an answer as the same question about marginalization.
;;
;;

;; ## Bayes' Rule
;;
;; <!-- TODO: make it clear that the issue here is dividing things between the observed and latent RVs -->
;;
;; Combining the definition of conditional probability with the chain
;; rule, we end up with an important special-case law of probability
;; known as *Bayes' Rule*:
;;
;; $$\Pr(H=h \mid D=d)=\frac{\Pr(D=d,  H=h)}{\sum_{h'\in H} \Pr(D=d \mid H=h')\Pr(H=h')}=\frac{\Pr(D=d \mid H=h)\Pr(H=h)}{\sum_{h'\in H} \Pr(D=d \mid H=h')\Pr(H=h')}=\frac{\Pr(D=d \mid H=h)\Pr(H=h)}{\Pr(D=d)}$$
;;
;; Note that this is just the definition of conditional probability with
;; $\Pr(D=d \mid H=h)P(H=h)$ substituted for $\Pr(D=d, H=h)$ via the
;; chain rule.  Bayes' rule is often written in this form with $H$
;; standing for a random variable representing some hypothesis, that is the latent variables in our model. $D$ stands for  *data* and represents the
;; observed variables in our model.
;;
;; The term $\Pr(H=h \mid D=d)$ is known as the *posterior probability*
;; of the hypothesis given the data. The term $\Pr(H=h)$ is known as
;; the *prior probability* of the hypothesis.  The term $\Pr(D=d \mid
;; H=h)$ is known as the *likelihood of the hypothesis* (or less
;; correctly, but still often, likelihood of the data), and the term
;; $\sum_{h'\in H} \Pr(D=d \mid H=h')\Pr(H=h')$ is known as the
;; *evidence* or, more often, *marginal likelihood* of the data. Note that
;; $\sum_{h'\in H} \Pr(D=d \mid H=h')\Pr(H=h')=\Pr(D)$, so this
;; denominator is just the marginal probability of the data
;; (marginalizing over all hypotheses).
;;
;; In the context of one of our joint models, we typically think
;; of the variable $H$ as containing all of the latent random variables
;; in our model, while the variable $D$ contains all of the observed
;; random variables. For instance, $H =\theta$ and $D = X^{(1)}, \dots, X^{(N)}$ in our biased coin model.
;;
;; There is nothing particularly special about Bayes'rule. It is just a straightforward application of some basic laws of probability theory. As we have discussed, which random variables in a model are seen as latent and which are seen as observed just depends on the application of our model. However, Bayes' rule does provide a convenient way of talking about different parts of our model: the prior over latent variables; the posterior over those same variables; the likelihood of the latent variables   given the observed; and the marginal likelihood of those variables, marginalizing away the latent variables. We will often use this terminology going forward.
;;
;; There is an important ways of thinking about how to use Bayes' rule. It gives a law relating our prior beliefs about our latents to our beliefs about the latents after observing some variables. We can apply this any number of times to sequentially update our beliefs. Examining the rule again, we see that
;;
;;
;; $$\Pr(H=h \mid D=d)\propto \Pr(D=d \mid H=h)P(H=h)$$
;;
;; In other words, our posterior beliefs in our latents are proportional to our prior beliefs, reweighted by a term that corresponds to the probability that the latent generated the observed values. One again, we see how probability theory is  conservative in terms of belief updating.
;;
;; ## The Posterior Predictive Distribution
;;
;; Let $D^\prime$ be a random variable representing the unobserved *next* datapoint we might sample from our model. The posterior predictive distribution is defined by:
;;
;; $$\Pr(D^\prime=d^\prime \mid D=d) = \sum_H \Pr(D^\prime=d^\prime \mid H=h) \Pr(H=h \mid D=d),$$
;;
;; or expanding terms:
;;
;; $$\Pr(D^\prime=d^\prime \mid D=d) = \sum_H \Pr(D^\prime=d^\prime \mid H=h) \frac{\Pr(D=d \mid H=h)P(H=h)}{\sum_{h'\in H} \Pr(D=d \mid H=h')\Pr(H=h')}.$$
;;
;;
;; In other words, it is defined by computing the distribution of the next observing averaging over all values of the latent random variables, weighting each by their posterior probability.
;;

;; # Posterior for the Biased Coin Model
;;
;; Let's write down the posterior and posterior predictive for our biased coin model. Because of the simplicity of this model, there is an exact form for these distributions which we can derive with a little calculus.
;;
;; The joint probability of the observed and latent random variables for the model is:
;;
;; $$\Pr(\{x_i\}_{i=1}^N, \theta) = \Pr(\theta)\prod_i^N \Pr(x_i|\theta)$$
;;
;; ## Posterior
;;
;; We first write down the posterior using the definition of conditional probability.
;;
;; $$\Pr(\theta|\{x_i\}_i^N) = \frac{\left[\prod_i^N \Pr(x_i|\theta)\right]\Pr(\theta)}{\Pr(\{x_i\}_{i=1}^N)}$$
;;
;; We then substitute the marginal in the denominator with its definition as an integral.
;;
;; $$\Pr(\theta|\{x_i\}_i^N) = \frac{\left[\prod_i^N \Pr(x_i|\theta)\right]\Pr(\theta)}{\int_\theta\left[\prod_i^N\Pr(x_i|\theta)\right]\Pr(\theta)d\theta}$$
;;
;; We now plug in the type of each of the probabilities above.
;;
;; $$ = \frac{\left[\prod_i^N \mathrm{Bernoulli}(x_i;\theta)\right]\mathrm{Beta}(\theta;\alpha,\beta)}{\int_\theta \left[\prod_i^N\mathrm{Bernoulli}(x_i;\theta)\right] \mathrm{Beta}(\theta;\alpha,\beta)d\theta}$$
;;
;; Replacing those types with densities, we get:
;;
;; $$ = \frac{\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)} \theta^{\alpha + N_h - 1}(1-\theta)^{\beta + N_t - 1}}{\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}\int \theta^{\alpha + N_h - 1}(1-\theta)^{\beta + N_t - 1}d\theta}.$$
;;
;; The constant term at the beginning which is a ratio of $\Gamma(\cdot)$ functions cancels.
;;
;; $$ = \frac{ \theta^{\alpha + N_h - 1}(1-\theta)^{\beta + N_t - 1}}{\int \theta^{\alpha + N_h - 1}(1-\theta)^{\beta + N_t - 1}d\theta}$$
;;
;; Taking the integral on the bottom gives us:
;;
;; $$ = \frac{\Gamma(N + \alpha + \beta)}{\Gamma(\alpha + N_h)\Gamma(\beta + N_t)} \cdot \theta^{\alpha + N_h - 1}(1-\theta)^{\beta + N_t - 1}.$$
;;
;; Here we used the fact that
;;
;; $$ = \frac{\Gamma(x)\Gamma(y )}{\Gamma(x+y)} = \int_{\Theta} \theta^{x- 1}(1-\theta)^{y - 1} d\theta.$$
;;
;;
;; This last expression is just the definition of a Beta distribution with new parameters!
;;
;; $$ = \mathrm{Beta}(\theta; \alpha + N_h, \beta + N_t)$$
;;
;; Interestingly, we see that the posterior for our model, that is the distribution that quantifies our uncertainty over the parameter $\theta$ after observing data, is another beta distribution, with adjusted parameters.
;;
;; > What is the interpretation of the new beta parameters.
;;
;;
;; ## Posterior predictive
;;
;; The posterior predictive distribution asks what is the distribution over next possible sampled observed values, given the preceding observed values. We compute it by averaging over the posterior.
;;
;; $$\Pr(x^{(N+1)}=h|\{x\}_i^N) = \int \Pr(x_{N+1}=h|\theta)\Pr(\theta|\{x\}_i^N)d\theta$$
;;
;; We substitute in the defintion of each of the pieces above.
;;
;; $$ = \int \theta \cdot \theta^{\alpha + N_h-1}(1-\theta)^{\beta+N_t-1} \frac{\Gamma(N + \alpha + \beta)}{\Gamma(\alpha + N_h)\Gamma(\beta + N_t)} d\theta$$
;;
;; Next we move the constant term outside of the integral.
;;
;; $$ = \frac{\Gamma(N + \alpha + \beta)}{\Gamma(\alpha + N_h)\Gamma(\beta + N_t)} \int \theta^{\alpha + N_h}(1-\theta)^{\beta+N_t-1}d\theta$$
;;
;; We now take the integral and get a product of two ratios involving the gamma function.
;;
;; $$ = \frac{\Gamma(\alpha+\beta+N)}{\Gamma(\alpha+N_h)\Gamma(\beta+N_t)} \cdot \frac{\Gamma(\alpha+N_h+1)\Gamma(\beta+N_t)}{\Gamma(\alpha+\beta+N+1)}$$
;;
;; Simplifying a bit
;;
;; $$ = \frac{\Gamma(\alpha+\beta+N)}{\Gamma(\alpha+\beta+N+1)} \cdot \frac{\Gamma(\alpha+N_h+1)}{\Gamma(\alpha+N_h)}$$
;;
;; Using some properties of the Gamma function (namely that it is like the factorial) we can simplify the above to
;;
;; $$ = \frac{\alpha+N_h}{\alpha+\beta + N}.$$
;;
;; > What is our predicted probability of heads? How does it take into account the data we have observed?

;; # Rejection Sampling
;;
;; Above, we asked what the sampling equivalent to conditioning was. There are many possible anwers to this question, many of which we will explore in this course. However, probably the simplest and easiest to understand is *rejection sampling*.
;;
;; Rejection sampling is a form of "guess and check"&mdash;we simply sample from the prior and then check to see if our conditioner is met. If it is, we return our sample, otherwise, we try again until we get a sample that matches the conditioner.
;;
;;

;; ```julia

;; # This function iterates over a ChoiceMap
;; function iter_deep(c::Gen.ChoiceMap)
;;   Iterators.flatten([
;;       Gen.get_values_shallow(c),
;;       (Dict(Pair{Any, Any}(k => kk, vv) for (kk, vv) in iter_deep(v))
;;        for (k, v) in Gen.get_submaps_shallow(c))...,
;;   ])
;; end

;; # Check if the conditions hold
;; function check_conditions(trace, constraints)
;;     for (name, value) in iter_deep(constraints)
;;         if trace[name] != value return false end
;;     end
;;     return true
;; end

;; function rejection(generative_model, arguments, constraints)
;;     t=Gen.simulate(generative_model,arguments)
;;     if check_conditions(t,constraints)
;;         t
;;     else
;;         rejection(generative_model, arguments, constraints)
;;     end
;; end

;; observations = Gen.choicemap()
;; observations[:x => 1] = true
;; observations[:x => 2] = true

;; valid_trace=rejection(flip_biased_coin, (3,), observations)
;; Gen.get_choices(valid_trace)

;; ```julia
;; num_flips=3
;; observations = Gen.choicemap()

;; histogram([rejection(flip_biased_coin, (num_flips,),
;;             observations)[:θ] for _ in 1:100000])

;; ```julia
;; num_flips=3
;; observations = Gen.choicemap()

;; for i in 1:(num_flips-1)
;;     observations[:x => i] = true
;; end

;; histogram([rejection(flip_biased_coin, (num_flips,),
;;             observations)[:θ] for _ in 1:100000])
;; ```

;; ```julia
;; num_flips=5
;; observations = Gen.choicemap()

;; for i in 1:(num_flips-1)
;;     observations[:x => i] = true
;; end

;; histogram([rejection(flip_biased_coin, (num_flips,),
;;             observations)[:θ] for _ in 1:100000])

;; ```julia
;; num_flips=30
;; observations = Gen.choicemap()

;; for i in 1:(num_flips-1)
;;     observations[:x => i] = true
;; end

;; histogram([rejection(flip_biased_coin, (num_flips,),
;;             observations)[:θ] for _ in 1:100000])
;; ```

;; Recall that a correct sampler must return values at a rate which is proportional to their true density. Thus, for rejection sampling to be a correct posterior inference algorithm, it must return latent variable values proportional to their posterior probability. Here is an informal argument that rejection sampling is correct.
;;
;; Suppose that we are trying to represent a conditional distribution $\Pr(\theta | X=\texttt{true})$. In rejection sampling we sample from the joint distribution $\Pr(\theta, X)$ and only keep samples that satisfy the condition, that is, $\Pr(\theta, X=\texttt{true})$.
;;
;; If we ignore the value of $\theta$ and only insist that $X=\texttt{true}$ the proportion of samples we keep will converge to $\Pr(X=\texttt{true})$. In other words, if
;; $N$ is the number of samples, and $N_{X=\texttt{true}}$ was the number of samples where $X$ is $\texttt{true}$ then $\frac{N_{X=\texttt{true}}}{N}$ will tend to $\Pr(X=\texttt{true})$ as $N$ gets large.
;;
;; However, instead of ignoring our sampled values of $\theta$, we return it. Consider a particular value $\breve{\theta}$, where the breve indicates some particular value of the the variable. Let's let the number of samples where $\theta$ is equal to this value and the condition is met be  $N_{\breve{\theta},X=\texttt{true}}$. Out of our overall samples returned the (sub)proportion where $\theta=\breve{\theta}$ is given by.
;;
;;
;; $$\frac{N_{\breve{\theta},X=\texttt{true}}}{N_{X=\texttt{true}}}$$
;;
;;
;; Note that this is equal to
;;
;;
;; $$\frac{\frac{N_{\breve{\theta},X=\texttt{true}}}{N}}{\frac{N_{X=\texttt{true}}}{N}}$$
;;
;; which will tend towards
;;
;; $$\frac{\Pr(\breve{\theta},X=\texttt{true})}{\Pr(X=\texttt{true})} = \Pr(\breve{\theta} \mid X=\texttt{true})$$
;;
;; as $N \rightarrow \infty$. And, of course, this is just our definition of conditional probability (in the discrete case).
;;
;; In cases where our condition has probability greater than $0$ and our generative model halts with probability $1$, rejection sampling is guaranteed to be a correct sampling algorithm.
;;
;; > What problems do you see with the rejection sampling approach to inference?
;;
;;

;; # The Cost  of Rejection Sampling
;;
;; Just how long will it take our rejection sampling algorithm to "hit" the observed dataset? To give an answer to this question, it is useful to first introduce the concept of the *geometric distribution*.

;; ## Geometric Distributions
;;
;; The *geometric distribution* with parameter $\theta$ is the distribution over numbers of  flips of a coin with bias $\theta$ until the first flip comes up heads.  It can be expressed with the following generative process.
;;

;; ```julia
;; @gen function geometric(θ)
;;    if bernoulli(θ)
;;         return(1)
;;     else
;;         1+geometric(θ)
;;     end
;; end;

;; histogram([geometric(0.5) for _ in 1:1000], label=false)
;; ```

;; Let's take a moment to consider this generative process. The generative process can be thought of as walking along the natural numbers
;; flipping a coin, with probability $\theta$ at each natural number. If
;; the coin comes up `true` at natural number $k$ then we stop
;; and return  $k$. Otherwise, we continue and repeat the
;; process at $k+1$. Thus, the probability distribution over integers
;; defined by this process is given by the following expression
;; (recalling that $1-\theta$ is the probability of **not** stopping at a
;; particular $k$).
;;
;; $$p(k;\theta) = \theta (1-\theta)^{(k-1)}$$
;;
;; The
;; geometric distribution is a distribution on the natural numbers
;; $\mathbb{N}$ which is parameterized by a fixed *success probability*
;; $\theta$.
;;
;; At this point, something may concern you. The set of probabilities characterizing a distribution
;; must sum to $1$. But now we have defined the geometric distribution
;; over the countably infinite set $\mathbb{N}$. Does it sum to $1$?
;;
;; Recall that a series of the form
;;
;; $$a + ar + ar^2 + ar^3 + ...$$
;;
;; is called a *geometric series*. When $|r|<1$, the sum of a such a geometric series is given by the following expression:
;;
;; $$a + ar + ar^2 + ar^3 + ...  = \sum_{k=1}^\infty ar^{(k-1)} = \frac{a}{1-r}, \
;; \text{for} |r|<1.$$
;;
;; For the geometric distribution, we set $a=\theta$ and $r=(1-\theta)$ and the
;; terms of this sum represent the probabilities of the outcomes at each
;; natural number $k$.
;;
;; $$\sum_{k=1}^\infty \theta (1-\theta)^{(k-1)}$$
;;
;; in other words
;;
;; $$\theta + \theta(1-\theta) + \theta(1-\theta)^2 + \theta(1-\theta)^3 + ...$$
;;
;; The sum of this series is
;;
;; $$\frac{\theta}{1-(1-\theta)} = \frac{\theta}{\theta} = 1$$
;;
;; Thus, this sequence sums to $1$, and thus we have defined a
;; well-formed probability distribution over the natural numbers.
;;

;; ## Expectation of a Geometric Distribution
;;
;; Let's use our definitions above to compute the expectation of a geometric distribution, that is, the expected number of coing flips until we halt.
;;
;; $$K \sim \mathrm{geometric}(\theta)$$
;;
;; Using the definition of expectations and our expression for the sum of a geometric series (several times) we have:
;;
;; $$\begin{align}\mathbb{E}[K] &= \sum_{k=1}^\infty k (1-\theta)^{k-1}\theta  \\
;; \mathbb{E}[K] &= \theta \left[ \sum_{k=1}^\infty  (1-\theta)^{k-1} + \sum_{k=2}^\infty  (1-\theta)^{k-1} + \sum_{k=3}^\infty  (1-\theta)^{k-1} +\dots \right] \\
;; \mathbb{E}[K] &= \theta \left[ \sum_{k=1}^\infty  (1-\theta)^{k-1} + (1-\theta) \sum_{k=1}^\infty  (1-\theta)^{k-1} + (1-\theta)^2 \sum_{k=1}^\infty  (1-\theta)^{k-1} +\dots \right] \\
;; \mathbb{E}[K] &= \theta \left[ 1/\theta + (1-\theta)/\theta + (1-\theta)^2/\theta + \dots \right] \\
;; \mathbb{E}[K] &= 1 + (1-\theta) + (1-\theta)^2 + \dots \\
;; \mathbb{E}[K] &= 1/\theta \\
;; \end{align}$$
;;
;;
;; In other words, the expected number of coin tosses of a biased coin with weight $\theta$ before you see the first `true` value is $1/\theta$. To put that in perspective, if the probabilty $\theta$ is rational, it can be written as $1/n$ for some positive integer $n$ and this the expected number of coin tosses is $n$.

;; # Rejection Sampler as a Geometric Distribution
;;
;; If our conditioner is $X=x$, then the marginal probability of the condition $\Pr(X=x)$ represents the percentage of the time we should expect a  random sample from the joint distribution to be consistent with the condition. Our arguments above showed then that the expected number of samples we should expect to take, on average, before we see one which satisfies the conditioner is $1/\Pr(X=x)$. How does this play out in practice?

;; ```julia
;; observations = Gen.choicemap()

;; for i in 1:10000
;;     observations[:x => i] = true
;; end

;; valid_trace=rejection(flip_biased_coin, (20000,), observations)
;; Gen.get_choices(valid_trace)
;; ```
