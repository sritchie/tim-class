;; # Modeling by Simulation

;; In this course, we will study a particular approach to modeling problems of
;; cognition, intelligence, and statistical inference&mdash;*modeling by
;; simulation*. This approach is also called the *generative approach* and
;; sometimes identified with *Bayesian statistics*. In this approach, we model
;; the world by simulating a simplified version of it.
;;
;; To connect our simulation based models with real-world phenomena, it is
;; critical that we correctly handle *uncertainty*. Uncertainty in modeling
;; arises from a number of sources. We might find that when we go to measure the
;; real world phenomena that we wish to model, our measurements are *noisy*. It
;; might also be the case that simplifications in our model demand that the
;; models be able to handle some amount of uncertainty in order to cover all
;; possible observed datasets. Many researchers also believe that uncertainty is
;; an inherent part of the world in which we live. Whatever the sources of
;; uncertainty, we must capture it to make models that are scientifically
;; useful.
;;
;; Our models should be powerful enough to capture a wide variety of different
;; kinds of phenomena from scientific theories of the physical universe to
;; particular sets of assumptions used in statistical analyses, to models of how
;; people think and act. The most general modeling idiom we have to date is that
;; of the *programming language*. All sufficiently expressive programming
;; languages are equivalent (to Turing machines) and by the Church-Turing
;; thesis, these capture the very notion of *effective computability*. To
;; provide the ability to handle uncertainty, we merely need to add in some
;; probabilistic primitives. Let's look at some examples in *Gen*. First, let's
;; load the Gen language.

;; # Generative Functions: Coin Flipping
;;
;; The simplest simulation based model, and a standard example for Bayesian
;; statistics, is a coin tossing model. Let's give ourselves a fair coin to make
;; use of. This is known as a *Bernoulli* distribution with parameter $.5$.

;; ```julia
;; bernoulli(0.5)
;; ```

;; In order to make this useful to the Gen interpreter, we will need to wrap it
;; in a *generative function*. A generative function is a special kind of
;; function that we can use to define probability distributions. We will see
;; much more about these later. For now, just note that a generative function
;; looks like a normal function but it is passed to the `@gen` macro and allows
;; some special syntax.

;; ```julia
;; @gen function flip()
;;   x  ~ bernoulli(0.5)
;; end
;; ```

;; This example makes use of the special Gen syntax `val ~ distribution()`. This is called the *sampling operator* and is the heart of how Gen works. The syntax of this operator is modeled on *squiggle notation* from probability theory and statistics, where we often see statements like:
;;
;; $$x \sim \mathrm{Bernoulli}(0.5)$$
;;
;; This is read as the random variable $x$ *is distributed as* a Bernoulli random variable with parameter (probability) $0.5$. But it can also be read as the *value* of $x$ is *sampled from* a Bernoulli distribution with probability parameter $0.5$. The sampling operator in Gen has many special properties which we will look at below.
;;
;; Note that unlike most functions you are familiar with, this function takes no arguments. In functional programming, such functions are called *thunks*. While a deterministic function that takes no arguments must have a constant return value, when our computation can contain  probabilistic choices, this is no longer true.
;;
;; Now we can call this function to sample different fair coin flips. Here we create an array with the output of `flip` two times.
;;

;; ```julia
;; flip()
;; ```

;; A model which gives an explicit formula for sampling from some distribution
;; is called a *generative model*. We say that such models are run in the
;; *forward direction*&mdash;the reasons for this will become clear when we
;; introduce inference via conditioning later. In Gen, there is an operator
;; `simulate` which explicitly runs such models in the forward direction. This
;; function additionally does some book keeping which will become important for
;; understanding how to do inference in probabilistic programming languages.

;; ```julia
;; Gen.simulate(flip,())
;; ```

;; Simulate takes two arguments. First, it takes the generative function you
;; wish to simulate. Second, it takes the tuple of arguments that should be
;; passed to that function when it is run.
;;
;; Note that unlike a simple call to `flip`, simulate returns a complex *trace*
;; object. We will learn much more about these traces and the information they
;; contain later.
;;
;; # Distributions and Random Variables

;; Before moving on, it is worth saying something about the meaning of some of
;; the terms we will use in this course. Intuitively a *distribution* is just a
;; set of possible outcomes called a *sample space* together with some way of
;; assigning numbers between $0$ and $1$ to those outcomes such that when you
;; add together all of the the probabilities they sum to $1$.
;;
;; Unfortunately, when the sample space is continuous (i.e., it as an
;; uncountable number of outcomes), such as in our beta example above, simple
;; intuitions break down and we have to be careful about the probabilities that
;; we assign to outcomes and sets of outcomes.
;;
;; To give a firm foundation to probability that avoids these issues, we must
;; use *measure theory*. In the measure-theoretic approach to probability
;; theory, what we informally call a distribution is formalized as a
;; *probability space* which is a triple $\langle \Omega,\mathcal{F}, \mathbb{P}
;; \rangle$ where $\Omega$ is the sample space (for instance, `{true,false}` for
;; our coin flipping example), $\mathcal{F}$ is the *event space* which is just
;; a collection of subsets of $\Omega$ that satisfy certain properties which
;; make $\mathcal{F}$ a $\sigma$-algebra and $\mathbb{P}: \mathcal{F}
;; \rightarrow [0,1]$ is the probability measure which assigns probabilities to
;; events in $\mathcal{F}$ (i.e., it assigns probabilities to sets of outcomes
;; in $\Omega$). The $\sigma$-algebra must satisfy a number of properties to
;; make $\langle \Omega,\mathcal{F}\rangle$ a *measurable space* and the
;; function $\mathbb{P}$ must assign a value between $0$ and $1$ to all elements
;; of $\mathcal{F}$ and $1$ to all of the outcomes in $\Omega$ taken together.
;;
;; Measure theory is needed to define coherent probability distributions on
;; continuous sets. When we are talking about discrete distributions (such as a
;; coin), then we can make use of a simpler definition of a probability
;; distribution that consists of two parts $\langle \Omega,p \rangle$, where
;; $\Omega$ is once again the sample space and $p$ is the *probability mass
;; function* which assigns probabilities that is, numbers between $0$ and $1$,
;; to elements of $\Omega$ such that $\sum_{\omega \in \Omega} p(\omega) = 1$.
;; Note that the domain of a probability mass function is the sample space,
;; **not** the $\sigma$-algebra, unlike a probability measure such as
;; $\mathbb{P}$. Of course, even for discrete distributions there exists a
;; formal probability space $\langle \Omega, \mathcal{F}, \mathbb{P} \rangle$.
;; Usually, in this case, $\mathcal{F}$ is just the power set of $\Omega$,
;; $2^{\Omega}$.
;;
;; > What is the relationship between a probability mass function $p$ and its
;; > corresponding probability measure $\mathbb{P}$?
;;
;; All of the continuous distributions that we will use in this course can also
;; be specified using a *probability density function* which has the sample
;; space as a domain. In fact, in measure theory, probability mass functions on
;; discrete spaces can be interpreted as particular kinds of densities (with
;; respect to the counting measure). Thus, throughout this course, we will refer
;; to the functions which define the probability of points in our sample spaces
;; as *probability density functions* (PDFs) or just *densities*.
;;
;; > What is the general relationship between a probability density function and
;; > its corresponding probability measure?
;;
;; Another important concept in probability theory is that of a *random
;; variable*. Intuitively, a random variable is place holder which takes on
;; values according to some distribution. Formally, a random variable is just a
;; measurable function from a distribution to another measurable space. This
;; definition is not usually particularly intuitive at first. From a
;; probabilistic programming perspective we can think of a random variable as
;; the **use** of a generative function at a particular place in the evaluation
;; of a program. For instance, `flip` is a distribution. However, in the program
;; below there are $3$ random variables.

;; ```julia
;; [flip(),  flip()]
;; ```

;; Although both random variables use the same underlying distribution, that is,
;; they are calls to the same generative function, they are distinct random
;; variables and can take on different values in the execution of a program.
;;
;; <!-- We have an underlying definition of the `flip` generative function which
;; represents the underlying probability space for this distribution. We can now
;; formalize the meaning of this program by defining three functions. The first
;; two function simply map from the underlying implementation of `flip` to each
;; of the instances in this program while the third constructs an array
;; containing these two functions.-->
;;
;; As we saw above, a critical feature (perhaps the fundamental feature) of Gen
;; is that it let's us name particular random variables using the sampling
;; operator.

;; ```julia
;; @gen function flip_two()
;;     [{:xs => i} ~ flip() for i in 1:2]
;; end

;; t=Gen.simulate(flip_two,())
;; Gen.get_choices(t)
;; ```

;; # Probability Notation
;;
;; Since probability is a tool used across mathematics, engineering, and the
;; sciences, there is an array of different notations and conventions used for
;; expressing probabilistic concepts in different contexts. Here we introduce
;; some of these.
;;
;; Often, random variables are written with upper case letters like $X$. When
;; $X$ is a random variable distributed according to some known distribution,
;; such as Bernoulli, we often write the following.
;;
;; $$X \sim \mathrm{Bernoulli}(\theta)$$
;;
;; As we saw above, this is read $X$ *is distributed as* a Bernoulli
;; distribution with *parameter* (i.e., `weight`) $\theta$. As we noted, it is
;; also sometimes read $X$ is *sampled from* a Bernoulli distrinbution with
;; parameter $\theta$.
;;
;; Another standard notation is to use $\Pr(X=x)$ to refer to the probability
;; that random variable $X$ takes on value $x$.
;;
;; $$\Pr(X=\texttt{true}) = \theta$$
;;
;; The notations $X=\texttt{true}$ and $\Pr(X=\texttt{true})$ are *logical*.
;; These are statements about one of outcomes in the range of the random
;; variable, and should not be confused with functions such as a PDFs and PMFs
;; which are often referred to with a lower case $p$ or $q$, for example,
;;
;; $$\Pr(X=\texttt{true}) = p_{\mathrm{Bern}}(\texttt{true}) = \theta$$
;;
;; Sometimes we will write $\Pr(X) = \{p(x) \mid x \in X\}$ to refer to the
;; whole distribution over the random variable $X$ or to the probability
;; evaluated at specific but unspecified values of $X$, $\Pr(X) := \Pr(X=x)$.

;; # Sampling as a Representation for Probability Distributions
;;
;; When we first learn probability, we are typically introduced to various
;; distribution, such as the *Bernoulli*, *beta*, or *Gaussian* distributions,
;; via their densities&mdash;functions which take points in the sample space of
;; the distribution and return the density or probability at that point, written
;; as $p(x)$. In this course, when we have a function which maps from a point in
;; a sample space to a density, we refer to this function as a *scorer* which
;; returns a *score*.
;;
;; Although we may not think about it when we are first exposed, working with
;; such density functions implicitly answers a question we may not have realized
;; we are asking:
;;
;; > From a computational point of view, what is the representation of a
;; > probability distribution?
;;
;; As we will see throughout the course, a probability distribution is a complex
;; object and we may wish to ask many different questions about it. When we wish
;; to ask and answer computational questions about a structure, we must decide
;; how to *represent* it in the computer. We shall see that many probability
;; distributions which are easy to conceptualize are difficult to represent
;; efficiently.
;;
;; One of the ideas of this course is that certain questions we might like to
;; answer with respect to a probability distribution are easiest to represent
;; using *samplers*&mdash;functions which, when called, draw a point from the
;; sample space of a probability distribution.
;;
;; Before moving on, it is useful to consider how samplers can let us answer
;; questions about distributions. Fundamentally, what they allow us to do is
;; build *histograms* of distributions.
;;

;; ```julia
;; histogram([sum([flip()  for i in 1:1000]) for i in 1:100])
;; ```

;; Of course, the more samples we take, the fine-grained and more accurate our histogram becomes (but the longer it takes to build the histogram).

;; ```julia
;; histogram([sum([flip()  for i in 1:1000]) for i in 1:1000000])
;; ```

;; The justificiation of working with samplers, comes from a mathematical
;; theorems known as the *law of large numbers* (actually there are many
;; versions of this law). Understanding the law of large numbers also helps us
;; understand a common, confusing notation which is used to express samplers.
;;
;; First, we need the concept of an expectation.
;;
;; ## Expectations
;;
;; A fundamental concept in probability theory is the *expected value* or
;; *expectation* of a random variable. Expectations are just a generalization of
;; the idea of a *mean* or *average* to arbitrary (non-uniform) probability
;; distributions.
;;
;; Suppose I have a set of numbers, $X$. How do I compute the mean or average of
;; this set of numbers?
;;
;; $$\sum_{x \in X} \frac{1}{|X|}x$$
;;
;; I can also compute a mean or average for some function of each number, $f$.
;;
;; $$\sum_{x \in X} \frac{1}{|X|}f(x)$$
;;
;; The mean assumes that each value is equally weighted (hence the
;; $\frac{1}{|X|}$ term). Instead of $X$ being a set, let's assume that $X$ is a
;; discrete random variable with an associated probability mass function $p$ and
;; weight each term $f(x)$ by $p(x)$.
;;
;; $$\sum_{x \in X} p(x)f(x)$$
;;
;; This quantity is known as the *expected value* or *expectation* of the
;; function $f$ with respect to the distribution $p$. When $X$ is a continuous
;; random variable, with an associated density function $p$ we use integration,
;;
;; $$\int p(x)f(x)dx.$$
;;
;; We often write the expectation with the following notation.
;;
;; $$\mathbb{E}_{X \sim p} [f(X)] = \mathbb{E}_{p(x)} [f(X)] = \int p(x)f(x)dx$$
;;
;; When it is clear which distribution we are taking expectations with respect
;; to, we often write just
;;
;; $$\mathbb{E} [f(X)] = \int p(x)f(x).$$
;;
;; Expected values are linear (in fact, convex) combinations of
;; (some function) of the values in the sample space of some random
;; variable. Thus, the expectation is *linear* meaning that the following
;; properties hold in general.
;;
;; $$\mathbb{E} [X + Y] = \mathbb{E} [X] + \mathbb{E} [Y]$$
;;
;; and
;;
;; $$\mathbb{E} [aX] = a\mathbb{E} [X] $$
;;
;; for arbitrary constant $a$.
;;
;; ## Law of Large Numbers
;;
;; Suppose that we are interested in the expectation of some function $f$ under
;; some distribution $p(x)$.
;;
;; $$\mathbb{E}_{p(x)}[f(X)]$$
;;
;; The law of large numbers tells us that this can be approximated by drawing
;; $K$ samples from $p$, and averaging their values under $f$, that is, we can
;; draw some samples, apply $f$ and take the resulting normal mean
;;
;; $$\mathbb{E}_{p(x)}[f(X)] \approx \sum_{i=1}^K \frac{1}{K} f(x_i), \quad x_i \sim p$$
;;
;; with the approximation getting better as $K$ increases. Thus, sampling is
;; theoretically justified whenever we want to answer a question about a
;; distribution that can be formulated as an expectation.
;;
;; We will see many complex approaches to sampling expressed using this notation.
;;
;; ## Expectations and Probabilities.
;;
;; We suggested above that an intuitive way of thinking about a sampler was in
;; terms of a histogram, which approximates a full distribution. But the law of
;; large numbers relates samplers to expectations of particular functions of a
;; distribution. What is the relationship between the two perspectives?
;;
;; Consider the special case where the function $f$ that we want to approximate
;; is an *identity function* $\mathbb{1}_{x^\prime}(x)$ which returns $1$
;; whenever the input $x$ is exactly equivalent to the value $x^\prime$, and $0$
;; otherwise (this is referred to as a *Dirac $\delta$* measure in measure
;; theory, and written $\delta_{x^\prime}$). So we wish to approximate this
;;
;; $$\mathbb{E}_{p(x)}[\mathbb{1}_{x^\prime}(X)] \approx \sum_{i=1}^K \frac{1}{K} \mathbb{1}_{x^\prime}(x_i), \quad x_i \sim p$$
;;
;; Note that in this sum, all of the outcomes that aren't exactly equal to
;; $x^\prime$ drop out, and so the result will just be the number of times we
;; sampled $x^\prime$ divided by the number of samples
;;
;; $$ \mathbb{E}_{p(x)}[\mathbb{1}_{x^\prime}(X)]\approx \frac{\;;(x_i=x^\prime)}{K}$$.
;;
;; Intuitively, this is an estimate of $p(x^\prime)$, at least when $X$ is a
;; discrete random variable.
;;
;; $$\mathbb{E}_{p(x)}[\mathbb{1}_{x^\prime}(X)] = p(x^\prime) \approx \sum_{i=1}^K \frac{1}{K} \mathbb{1}_{x^\prime}(x_i), \quad x_i \sim p.$$
;;
;; Since we can use sampling-based estimates of expectations to estimate the
;; probability of any outcome in our distribution, we can use it to do anything
;; we could do with a histogram. This clarifies the relationship between the two
;; perspectives.

;; # Biased Coin Flipping
;;
;; Now let's set up a slightly more complex problem, with more uncertainty built
;; in. We imagine that we are in a setting where we see the result of some
;; number $n$ coin flips; however, we think that the coin may be biased (bent or
;; weighted) in some way, but we don't know the weight in advance. How can we
;; write a generative process that captures this idea. First, let's give
;; ourselves a useful function called `repeatedly` that takes a thunk `f` and a
;; number of times to call it repeatedly as inputs and returns a list of the
;; return values of the $n$ calls to `f`.

;; ```julia
;; @gen function repeatedly(f,n)
;; xs = [{:xs => i} ~ f() for i in 1:n]
;; end
;; ```

;; We have used Julia's *array comprehension* syntax which looks like this
;; `[expression for element = iterable]` to create an array of output values.
;;
;; Note that here we used the sampling operator in a slightly more complex way
;; than before: `{:xs => i} ~ f()`. Here the syntax `{:xs => i}` gives a *name*
;; to each particular call to `f` in the repeat function. In Julia `=>` is the
;; operator that creates *pairs*.

;; ```julia
;; typeof(1 => 2)
;; ```

;; And prepending a `:` to a string makes a *symbol*.

;; ```julia
;; :f

;; typeof(:f)
;; ```

;; In general, we can name the return values of particular random choices using
;; the syntax `{expression} ~ f()` where `f` is some distribution (generative
;; function). If we wish to also assign the output of the sampling process to a
;; variable, we can use the syntax `x = {expression} ~ f()`. This sampling
;; operator syntax only works in the context of the `@gen` macro, that is,
;; inside of generative function definitions and is specific to Gen, not Julia.

;; ```julia
;; x = {:x} ~ flip()
;; ```

;; The syntax `x ~ f()` is shorthand for `x = {:x} ~ f()`; that is, it assigns
;; the sample value to the variable `x` and gives that particular sample the
;; name of the variable which is the symbol `:x`.
;;
;; Let's define a generative process for biased coins.
;;
;; First, to represent uncertainty over the weight of the coin, $\theta$, we
;; will assume that the coin's weight is itself drawn from some distribution. A
;; convenient distribution for this is the *beta* distribution. The beta
;; distribution is a distribution on the unit interval. Thus, it can be thought
;; of as a distribution on probabilities $p$ and, therefore, is useful for use
;; as a *prior distribution* on our coin weight.
;;
;; The beta distribution takes two parameters, often called $\alpha$ and
;; $\beta$. Here is a picture (from Wikipedia).
;;
;; <img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg"
;;      alt="Beta Distribution"
;;      width="300" height="300"/>
;;
;; The parameters of the beta distribution control it's shape. A $beta(1,1)$
;; distribution is *uniform* on the unit interval and we will make use of this
;; parameterization below.
;;

;; ```julia
;; beta(1,1)
;; ```

;; ```julia
;; @gen function flip_biased_coin(n)
;;     θ = {:θ} ~ beta(1,1)


;;     @gen function weighted_coin()
;;         x = {:x} ~ bernoulli(θ)
;;     end

;;     xs = {:xs} ~ repeatedly(weighted_coin, n)
;; end

;; n_flips=10
;; flip_biased_coin(n_flips)
;; ```

;; How does the proportion of zeros and ones affect you judgment about the
;; weight of the coin that was tossed?

;; # Traces
;;
;; Now we come to the heart of the bookkeeping that Gen does for generative
;; functions. Let's consider the `flip_two` function we defined above. When we
;; call this with simulate, it returns an [*execution
;; trace*](https://www.gen.dev/dev/ref/gfi/#Traces-1) object.

;; trace=Gen.simulate(flip_two,())
;; typeof(trace)

;; Trace objects record several different pieces of information about the
;; execution of a generative function. These include (but aren't limited to):
;;
;;  1. The arguments passed to the generative function.
;;  2. The random choices made in evaluating the generative function.
;;  3. The return value returned by the generative function.
;;  4. The score of the evaluation of the generative function (see below).
;;
;; By calling the `get_choices` function on the trace, we get a record of the
;; random choices that were executed when the generative function was called. We
;; see here an entry for each element in the list returned by `flip_two`. Since
;; we defined `flip_two` using another generative function, namely `flip` we
;; also see the place where the individual flips were named as `:x` displayed in
;; this trace.

;; ```julia
;; Gen.get_choices(trace)
;; ```

;; Computation traces are important because they give us a structured way to
;; talk about all of the random variables which were populated with values in
;; the course of execution of a program. Fundamentally, it is the trace objects
;; which give us a *complete* picture of all of the random choices that are made
;; when we run a model, as well as the arguments of the function, it's return
;; values, and the probability of that program execution (the *score* of the
;; trace).
;;
;; Let's look at the trace for the `flip_biased_coin` function we defined above.

;; ```julia
;; trace=Gen.simulate(flip_biased_coin, (5,))
;; Gen.get_choices(trace)
;; ```

;; Note that here, because `flip_biased_coin` itself was defined by composing
;; several generative functions, the resulting trace has a lot of embedded
;; structure.
;;
;; The set of random choices in the particular execution of a generative
;; function are stored in
;; a [`ChoiceMap`](https://www.gen.dev/dev/ref/choice_maps/;;Gen.ChoiceMap)
;; object and can be thought of as a tree built from pairs. We can address into
;; the choicemap directly by treating the trace object as a dict and giving a
;; path to the relevant node in the choicemap (as a sequence of pairs).
;;
;;

;; ```julia
;; trace[:θ]

;; trace[:xs=>:xs=>4=>:x]
;; ```

;; <!-- trace[:xs=>:xs=>4] -->
;;
;; Notice that in our traces above, there is quite a lot of indirection. This is
;; because our program makes use of multiple generative functions that each
;; assign names to random variables created in their definitions. Since these
;; functions call one another, the ultimate choicemap that stores all such
;; values can have many unncessary names.
;;
;; For example, we see that the `:xs` node in the trace immediately dominates
;; another `:xs` node and these nodes are redundant&mdash;they only exist
;; because we have named the calls inside of `repeatedly` with `{:xs => i}` and
;; then again named the call to `repeatedly`.
;;
;; We can avoid this with the splice naming operator, which looks like this `var
;; = {*} ~ f()`. This operator tells Gen to splice the result into the choicemap
;; of the calling function, rather than give it a separate name to the node
;; itself.

;; ```julia
;; @gen function flip_biased_coin(n)
;;     θ = {:θ} ~ beta(1,1)

;;     @gen function weighted_coin()
;;      x ~ bernoulli(θ)
;;     end

;;     xs = {*} ~ repeatedly(weighted_coin, n)
;; end


;; trace=Gen.simulate(flip_biased_coin, (5,))
;; Gen.get_choices(trace)
;; ```

;; We can see now that one of the `:xs` nodes has disappeared.
;;
;; These examples illustrates another point. The random variable associated with
;; the call to simulate above is a *compound random variable* built from
;; primitive random variables like `flip` and `repeat`. In general, in
;; probabilistic modeling we will often construct complex distributions from
;; simpler ones. We now turn to discussing how such complex *joint
;; distributions* can be constructed from pieces.

;; # Elementary and Compound Distributions and RVs
;;
;; We have now seen how to build several different kinds of distributions in Gen
;; and how these can be used to implement random variables in more complex
;; programs. We have also seen how particular random variables in a program can
;; be named using the sample operator and how the trace of a generative function
;; represents the set of random variables evaluated (i.e., for which a value was
;; sampled) during the execution of the function.
;;
;; There is an important distinction between different kinds of distributions
;; that we use in our programs. An *elementary distribution* is a distribution
;; that is atomic in the sense that sampling from it does not involve sampling
;; from other, more primitive, random variables.
;;
;; We have seen several examples of elementary distributions, such as `bernoulli`.

;; ```julia
;; bernoulli(0.1)
;; ```

;; Technically, `bernoulli` actually represents a *family* of distributions,
;; indexed by the weight parameter; that is, `bernoulli(0.5)` is a different
;; distribution from `bernoulli(0.3)`.
;;
;; By contrast with elementary distributions, *compound distributions* are those
;; that are built up from some combination of elementary distributions. We have
;; seen several examples of compound distributions, such as the distribution
;; defined by `flip_biased_coin`.
;;
;; When we use an elementary or compound distribution in a program, we can also
;; talk about particular random variable which make use of the distribution
;; being elementary or compound.
;;
;; In general, elementary random variables are instantiated at the leaves of
;; traces, while more complex random variables correspond to higher-level nodes
;; in the trace. In Gen, elementary random variables are sometimes called
;; *choices*.
;;

;; # Joint Distributions
;;
;; In probability theory, especially in Bayesian approaches, the fundamental
;; mathematical object of interest is the *joint distribution*. From the point
;; of view of probabilistic programming, we can think of the joint distribution
;; as the distribution over all possible execution traces of our generative
;; model. That is, it is the distribution over **all** random variables that are
;; instantiated during the execution of our program.
;;
;; Let's call the $i$th random variable in our execution trace $X_i$. The *joint
;; distribution* over the trace can be written as follows:
;;
;; $$\Pr(X_1, X_2, \dots, X_K)$$
;;
;; Where $K$ is the number of random variables in the trace. In fact, traces can
;; contain different numbers of random variables on different executions of the
;; same generative model, as we will see in later models in the course.
;;
;; What exactly is a joint distribution from a mathematical point of view?
;; First, let's consider the question of what the sample space $\Omega$ is for a
;; given joint distribution. In any given execution of a program there will be
;; some set of elementary random variables which will get instantiated and these
;; will be represented at the leaves of the correponding execution trace.
;; Without loss of generality, we can consider the sample space of a joint
;; distribution to be the Cartesian product of the sample spaces for the set of
;; elementary distributions that are instantiated as random variables in a
;; program execution. In other words, the joint sample space is the set of
;; tuples of the form $\langle \bar{X}_1=x_1, \dots, \bar{X}_K=x_K \rangle$
;; where the bar overset $\bar{X}$ means that a random variable instantiates an
;; elementary distribution. Consider the example of `flip_two`. This procedure
;; ultimately grounds out in two calls to the `bernoulli(0.5)` distribution.
;; Therefore, it's sample space consists of the set of all pairs of booleans:
;; $\{\texttt{true}, \texttt{false} \} \times \{\texttt{true}, \texttt{false}
;; \}$
;;
;;
;; Note that for practical probabilistic programs, we will generally make use of
;; many compound random variables in our generative models and inference
;; algorithms. It will often be important to name subtrees of the trace that
;; combine multiple elementary random variables. Nevertheless, elementary random
;; variables are the only sources of randomness in our programs; all other
;; computation is deterministic. Thus, a specification of just the random
;; choices made by the elementary random variables will suffice to completely
;; characterize the execution of our program and it is useful to think of our
;; sample space as being defined in terms of tuples of these objects.
;;
;; It is often convenient to visualize joint distributions as a table with one
;; dimension for each elementary random variable. Let's visualise a two
;; dimensional joint distribution for `flip_two` this way.
;;
;; |    $$\mathbf{X}_1\backslash \mathbf{X}_2$$  |`true`|`false`|
;; |---      |---|---|
;; | `true`  |  $$p(\texttt{true},\texttt{false})$$ |$$p(\texttt{true}, \texttt{true})$$|
;; | `false` |  $$p(\texttt{true}, \texttt{false})$$ |$$p(\texttt{true},\texttt{false})$$|
;;
;; Notice that the complexity of a joint distribution increases quickly as you
;; add values to each random variable and as you add random variables (i.e.,
;; dimensions in the table). For instance, for a sample space of size four, the
;; set of tuples over two random variables $\Pr(X_1,X_2)$ could have a different
;; probability for every single one of the $16$ different combinations of
;; outputs. For tuples over three random variables $\Pr(X_1,X_2, X_3)$ with a
;; sample space of size $4$ there could be a different probability for each of
;; the $64$ possible strings, and so on.
;;
;; <!-- What is the probability mass function associated with such a joint
;; distribution? In general, the probability mass function for a joint
;; distribution needs to specify a probability for every combination of values
;; possible for the random variables in the distribution.
;;
;; $$\Pr(\bar{X}^{(1)} = x^{(1)}, \bar{X}^{(2)} = x^{(2)})$$
;;
;; So when we construct a random variable out of two Bernoullis, there will be 4
;; possible *states* in the joint distribution, and when we construct random
;; variable out of three Bernoullis there will be 8 possible states in the
;; joint. Thus, the full probability mass function $p$ can potentially involve
;; specifying seven probabilities (eight minus one).
;; -->
;;
;; <!--
;;
;; Joint distributions are typically considered the fundamental object of
;; interest in a probabilistic model since they encode the maximum
;; possible amount of information about interactions between random
;; variables. Joint distributions can be very complex and can require
;; many parameters to specify completely.  Because of this, practical
;; probabilistic models almost always make simplifying assumptions that
;; make the representation of the joint distribution less complex, and
;; prevent us from having to specify a different probability for every
;; value of every combination of random variables in the table. In
;; particular, models make *conditional independence assumptions* about
;; random variables.  For example, in the case of the our `flip_two` model, the
;; probability mass function is given by the product of the probabilities
;; of each individual coin flips.
;;
;; $$\Pr(X^{(1)}, X^{(2)}) = \Pr(X^{(1)}) \times \Pr(X^{(2)})$$
;;
;; The coin flips in this simple model are *statistically
;; independent*. Statistical independence is a fundamental concept in
;; probabilistic modeling, and we will now consider it in more detail.
;; -->
;;
;; <!-- TODO: introduce geometric distributions -->
;;
;; <!--- TODO: conditional probabilities -->
;;
;; What about the probabilities associated with a particular outcomes in joint
;; distribution define by a generative function? To examine this question, we
;; need to introduce the notion of *statistical independence*.

;; # Independence
;;
;; Intuitively, *independence* means that the values that one random variable
;; takes on don't affect our beliefs about the values that another random
;; variable can take on. When sampling from one random variable, we are
;; **disallowed** from using the value that another as information that informs
;; how we draw our sample, independent random variable has taken on or,
;; equivalently, we do not **need** this information to take the sample.
;; Mathematically, this idea is captured in probability theory by the following
;; principle: If $A$ and $B$ are *independent random variables*, then we can use
;; the *product rule* of probability theory to compute the joint probability.
;;
;; $$\Pr(A=a, B=b)=\Pr(A=a)\times \Pr(B=b)$$
;;
;; More generally, if some set of random variables $X_1, \dots, X_K$ are
;; independent from one another then
;;
;; $$\Pr(X_1, \dots, X_K )= \prod_{i=1}^K \Pr(X_i)$$
;;
;; When the elementary random variables in a generative model are independent we
;; can use the product rule to calculate the probability of each *point* or
;; *state* in the joint distribution. Each such point is identified with exactly
;; one execution trace of the generative model.
;;
;;
;; This is can be illustrated with our `flip_two` model. In this model, each
;; coin flip is independent of the other flips. There are four states in the
;; joint, and their probabilities can be computed according to the product rule
;; as follows.
;;
;;
;; |    $$\mathbf{X}_1\backslash \mathbf{X}_2$$  |`true`|`false`|
;; |---      |---|---|
;; | `true`  |  $$p(\texttt{true})\times p(\texttt{false})$$ |$$p(\texttt{true})\times p(\texttt{true})$$|
;; | `false` |  $$p(\texttt{true})\times p(\texttt{false})$$ |$$p(\texttt{true})\times p(\texttt{false})$$|
;;
;;
;; We can see the score of a particular trace by using the `Gen.get_score`
;; function.

;; ```julia
;; trace=Gen.simulate(flip_two, ())
;; exp(Gen.get_score(trace))
;; ```

;; Note that is nearly always the case when working with probabilities in a
;; computational setting, we represent them in log space.
;;
;; Since our model uses a fair coin, all individual flip outcomes have
;; probability $0.5$ and, thus, every combination of two outcomes has
;; probability $0.25$.

;; # Conditional Probabilities
;;
;; In generative modeling, we build models from *conditional probability
;; distributions*. Conditional probabilities are written like this:
;;
;; $$\Pr(A=a\mid B=b)$$
;;
;; Such an expression is read as the probability that random variable $A$ takes
;; on value $a$ *given* or *conditioned on* that the random variable $B$ has
;; taken on the value $b$.
;;
;; Critically, the information to the right of the conditioning bar must be some
;; concrete information&mdash;an actual value that $B$ takes on. The conditional
;; distribution is a distribution over $A$ which is *indexed* by that particular
;; value of $B$; thus, there will be a separate distribution over $A$ for each
;; value $B=b$. We can think of the set of conditional distributions indexed by
;; $b$ as a dictionary which contains distributions over $A$ which are hashed on
;; values of $b$.
;;
;; More generally, we can consider the joint distribution on some set of random
;; variables $X_1, \dots, X_K$ conditioned on another set of random variables
;; $Y_1, \dots, Y_N$ taking on some values.
;;
;; $$\Pr(X_1, \dots, X_K \mid Y_1, \dots, Y_N)$$
;;
;; Here we are using the notational convention that $\Pr(X \mid Y)$ means
;; $\Pr(X=x \mid Y=y)$ for arbitrary $x$ and $y$.
;;
;; Conditional probabilities have several (equivalent) interpretations. We can
;; think of them as a sort of hypothetical reasonining: *What would the
;; probability of $A=a$ be if I knew that $B=b$ were true?* This is the most
;; common interpretation when we are considering a conditional probability
;; distribution which is not already given as part of our generative model. We
;; will work extensively with this interpretation of conditional probabilities
;; when we begin studying Bayesian inference.
;;
;; When a conditional distribution is specified as part of the definition of a
;; model, it has another, equivalent interpretation. In that case, we think of
;; the conditioner as providing specific information which *must* be present in
;; order to draw a sample.
;;
;; For example, in the `flip_biased_coin` each call to `bernoulli` must be
;; passed a value for `weight`. There is simply no other way to sample the
;; flip&mdash;if you didn't know the value of `weight`, you couldn't draw the
;; sample.
;;
;; When we discussed indepedence above we said that random variables were
;; independent if, when sampling from one random variable, we are **disallowed**
;; from using the value that another, independent random variable has taken on
;; or, equivalently, we do not **need** this information to take the sample. In
;; the case of a conditional distribution used as part of a model definition, we
;; are typically instead **required** to use the value of another random
;; variable, and in fact we absolute **need** this information to draw our
;; sample.
;;

;; # Conditional Independence
;;
;; Intuitively, *independence* means that the values that one random variable
;; takes on don't "affect" our beliefs about the values that another random
;; variable takes on. *Conditional independence* refers to the situation when
;; this is true when some additional information is available. When sampling
;; from one random variable we don't know the value the other one has taken on,
;; and don't need to, given this side information. Mathematically, this idea is
;; captured in probability theory by the following rule: If $A$ and $B$ are
;; *conditionally independent random variables given $C$*, then:
;;
;; $$\Pr(A=a, B=b \mid C=c)=\Pr(A=a \mid C=c)\times \Pr(B=b \mid C=c)$$.
;;
;; More generally, some set of random variables $X_1, \dots, X_K$ are
;; conditionally independent given another set of random variable $Y_1, \dots,
;; Y_N$ if
;;
;; $$\Pr(X_1, \dots, X_K \mid Y_1, \dots, Y_N)= \prod_{i} \Pr(X_i \mid Y_1, \dots, Y_N)$$
;;
;; The code inside a generative function that defines some generative model
;; specifies a sampling procedure that will sample each elementary conditional
;; distribution in some order given by the logic of the program. This fact
;; allows us to calculate the probability of each particular execution of the
;; program as the product of the probability of the values sampled by each
;; elementary conditional distribution. These executions are represented by
;; traces and are often referred to as *points* or *states* in the joint
;; distribution.
;;
;; This is can be illustrated with our `flip_biased_coin` model. In this model,
;; each coin flip is conditionally independent of the other flips given the
;; value of $\theta$.
;;
;; >  Are the random variables representing the individual flips independent
;; >  without $\theta$? Why or why not? Can you give an intutive argument?
;;
;; We can express the probability associated with traces of this model,
;; therefore, using the following product of elementary conditional
;; probabilities.
;;
;; $$\Pr(\theta, \{X_i\}_{i=1}^N)=\Pr(\theta) \prod_{i=1}^{N} \Pr(X_i \mid \theta)$$
;;
;; Consider a case where we flip our biased coin only twice after drawing our
;; `weight`. The probability distribution over the possible outcomes is now
;; given by the decomposition in the table below.
;;
;;
;; |    $$\mathbf{X}_1 \backslash \mathbf{X}_2$$  |`true`|`false`|
;; |---      |---|---|
;; | `true`  |  $$p(\theta)p(\texttt{true},\texttt{false} \mid \theta)$$ |$$p(\theta)p(\texttt{true}, \texttt{true} \mid \theta)$$|
;; | `false` |$$p(\theta)p(\texttt{false}, \texttt{true} \mid \theta)$$ |$$p(\theta)p(\texttt{false},\texttt{false} \mid \theta)$$|
;;
;;
;; When specifying a generative model, we always make use of such a
;; *factorization* of the model into elementary conditional probability
;; distributions. The factorization expresses the *conditional independence
;; assumptions* of the model.
;;
;; From a probabilitic programming perspective, such factorizations correspond
;; to the ordered sampling scheme which is specified by the generative function
;; that defines our model. The set of random choices made along the way is given
;; by the trace of the execution of this program.
;;
;; Thus, to reiterate, the score of a trace can be computed by taking the
;; product of all the elementary conditional distributions specifying each
;; random choice, in the order they are specified by the generative model.

;; ```julia
;; trace=Gen.simulate(flip_biased_coin, (5,))
;; exp(Gen.get_score(trace))

;; trace=Gen.simulate(flip_biased_coin, (1000,))
;; exp(Gen.get_score(trace))
;; # Gen.get_choices(trace)
;; ```

;; > Notice that as we generate more flips, the score of each trace goes
;; > down (on average). Why is this?
;;

;; # The Chain Rule and Conditional Independence
;;
;; An identity of fundamental importance in probability theory is the so-called
;; *chain rule*. The chain rule gives a way of decomposing a joint distribution
;; over several random variables into a sequence of conditional distributions
;; over individual random variables. For random variables $A$ and $B$, it can be
;; stated as follows.
;;
;; $$\Pr(A=a, B=b)=\Pr(A=a)\times \Pr(B=b \mid A=a) =\Pr(B=b)\times \Pr(A=a \mid B=b)$$.
;;
;; Note that this is an exact equality, for any pair of random variables, no
;; additional condition independency assumptions need to be made for this
;; identity to hold.
;;
;; In general, we can write
;;
;; $$\Pr(X_1, \dots, X_K)= \Pr(X_{\sigma(1)}) \prod_{i=2}^K \Pr(X_{\sigma(i)} \mid \{X_{\sigma(j)}\}_{j=1}^{i-1}).$$
;;
;; Here $\sigma(\cdot)$ is an arbitrary permutation of the indices on the
;; variables. In other words, we can pick any ordering we want for our
;; variables, and the rewrite the joint distribution by writing a product of
;; conditional distributions where each conditional is over just one random
;; variable conditioned on all the random variables that came earlier in the
;; sequence. For example,
;;
;; $$\Pr(X_1, \dots, X_K)= \Pr(X_1)\Pr(X_2 \mid X_1) \Pr(X_3 \mid X_1, X_2) \Pr(X_4 \mid X_1, X_2, X_3) \dots$$
;;
;; This rule, of course, also holds if we also condition on some other random
;; variables throughout.
;;
;; $$\Pr(X_1, \dots, X_K \mid Y_1, \dots, Y_N)= \Pr(X_{\sigma(1)} \mid Y_1, \dots, Y_N) \prod_{i=2}^K \Pr(X_{\sigma(i)} \mid \{X_{\sigma(j)}\}_{j=1}^{i-1}, Y_1, \dots, Y_N)$$
;;
;; The chain rule is useful, because it gives a way to think about how to
;; represent a complex joint distribution as a series of distributions which
;; consider each random variable "one at a time" in some order.
;;
;; It is also useful, because when we consider a particular generative model, we
;; can use it to see precisely which conditional independence assumptions our
;; model is making by comparing to the same ordering of variables in the
;; chain-rule decomposition of the joint. For instance, we noted above that the
;; probability of a particular execution trace for a call to
;; `flip_biased_coin(2)` might be given as follows.
;;
;; $$\Pr(\theta)\Pr(\texttt{true} \mid \theta)\Pr(\texttt{false} \mid \theta)$$
;;
;; The chain rule tells us that an exact represention of the full joint
;; probability, with no conditional independence assumptions could be given as
;; follows.
;;
;; $$\Pr(\theta,\texttt{true},\texttt{false}) = \Pr(\theta)\Pr(\texttt{true} \mid \theta)\Pr(\texttt{false} \mid \texttt{true}, \theta)$$
;;
;; By comparing the first and second expressions, we can see that our model
;; makes an important simplifying assumption compared to the full
;; joint&mdash;namely, it treats the coin flips as conditionally independent
;; given the weight of the coin.
;;

;; # The Full Biased Coin Flipping Model
;;
;;
;; Let's write out the full biased coin flipping model mathematically. First,
;; let's introduce the beta distribution in a bit more detail.

;; ## More on the beta distribution
;;
;; The beta distribution defines a distribution on the unit interval with
;; density given by
;;
;; $$p_{\mathrm{beta}}(\theta;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}.$$
;;
;;
;; Here we are using the convention that variables appearing after a semicolon
;; in a function definition refer to parameters of the function. The two
;; parameters in a beta distribution $\alpha$ and $\beta$ are known as
;; *pseudocounts*. They can be thought of as the imaginary number of times we
;; have seen outcome <tt>true</tt> and <tt>false</tt> before we draw any samples
;; from the distribution.
;;
;; How do the values of the two parameters influence the shape of the beta
;; distribution. We can understand this by remembering the following facts.
;;
;;  1. Increasing $\alpha$ will increase the probability of <tt>true</tt> and increasing $\beta$ will increase the probability of <tt>false</tt>.
;;
;;  1. The mean value of $\theta$, the probability of <tt>true</tt>, will be given by $\frac{\alpha}{\alpha+\beta}$.
;;
;;  2. When $\alpha=\beta=1$ the density will be uniform on the unit interval.
;;  3. For values of $\alpha$ and $\beta$ greater than $1$, the distribution will tend to be peaked around it's mean.
;;  4. For values of $\alpha$ and $\beta$ less than $1$, the distribution will tend to "throw" probability mass away from the mean towards the more extreme ends of the distribution.
;;
;; These behaviors can be seen in the following figure.
;;
;;
;; <img src="figures/Beta_distribution_pdf.svg"
;;      alt="Beta Distribution"
;;      width="500" height="500"/>
;;
;; In the mathematical expression for the beta density, the term
;; $\frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)}$ is the
;; normalizing constant, meaning it just makes the entire thing integrate to
;; $1$. Thus, it is the term $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ that does
;; the work of the distributon. We could write
;;
;; $$p_{\mathrm{beta}}(\theta;\alpha,\beta) \propto  \theta^{\alpha-1}(1-\theta)^{\beta-1}.$$
;;
;;
;; > Why is this way of expressing the beta distribution interesting or
;; > important?
;;
;;
;; Let's return to our biased coin model. We first repeat the generative
;; function for reference.

;; ```julia
;; @gen function flip_biased_coin(n)
;;     θ = {:θ} ~ beta(1,1)

;;     @gen function weighted_coin()
;;         x  ~ bernoulli(θ)
;;     end


;;     {*} ~ repeatedly(weighted_coin, n)
;; end

;; trace=Gen.simulate(flip_biased_coin, (5,));
;; Gen.get_choices(trace)
;; ```

;; We can express this model in squiggle notation as
;;
;; $$\theta \sim \mathrm{beta}(\alpha, \beta)$$
;; $$\{X_i\}_{i=1}^N \sim \mathrm{Bernoulli}(\theta).$$
;;
;; Second, the joint probability for this model can be expressed by the factored
;; product
;;
;; $$\Pr(\theta, \{X_i\}_{i=1}^N \mid \alpha, \beta) = \Pr(\theta \mid \alpha, \beta) \prod_{i=1}^N \Pr(X_i \mid \theta).$$
;;
;; where
;;
;; $$\Pr(\theta \mid \alpha, \beta) = p_{\mathrm{beta}}(\theta;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1} $$
;;
;; and
;;
;; $$\Pr(X \mid \theta)
;; = p_{\mathrm{Bern}}(X;\theta)
;; = \begin{cases}
;;   \theta   & \text{if }X=\mathtt{true} \\
;;   1-\theta & \text{if }X=\mathtt{false}
;; \end{cases}
;; $$
;;
;; Thus, the full model can be expressed as follows.
;;
;; $$\Pr(\theta, \{X_i\}_{i=1}^N \mid \alpha, \beta)=\frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha-1}(1-\theta)^{\beta-1}  \prod_{i=1}^N \theta^{X_i} (1-\theta)^{1-X_i}.$$
;;
;; Here we are using the convention that $X_i \in \{0,1\}$ instead of
;; $\{\mathtt{false},\mathtt{true}\}$. If we count the number of outcomes that
;; were true and false and let $N_{\texttt{T}}$ and $N_{\texttt{F}}$ represent
;; these quantities, we can rewrite the expression above as
;;
;; $$\Pr(\theta, \{X_i\}_{i=1}^N \mid \alpha, \beta)=\frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} \theta^{\alpha+ N_{\texttt{T}}-1}(1-\theta)^{\beta+N_{\texttt{F}}-1}.$$
;;
;;
;; Another convenient notation which we will see many times in this course is
;; *graphical model notation*.
;;
;; <img src="figures/coin-flipping-bn.jpg"
;;      alt="Coin Flipping PGM"
;;      width="100" height="100"/>
;;
;; Graphical model notation is a compact way of expressing the conditional
;; independence assumptions of a model as a graph with different kinds of nodes.
;; Constants, such as parameters like $\alpha$ and $\beta$ are drawn without a
;; surrounding node. *Unobserved* or *latent* random variables, such as $\theta$
;; in this model, are drawn in round nodes that are unshaded. *Observed* random
;; variables, such as $X$, are drawn as greyed out round nodes. Finally, this
;; example also uses *plate notation*. The *plate* under the $X$ node means that
;; we sample $X$ $N$ times and each sample is *conditionally independent and
;; identically distributed given* $\theta$, or i.i.d. given $\theta$. The arrows
;; represent the conditional independence assumptions of the model.
;;
;; > What would the graphical model look like for the full chain rule decomposition we discussed above?
;;
;; Note the close relation between the trace and the graphical model. The
;; graphical model displays all of the elementary conditional distributions used
;; to define our model and the dependencies between them. Thus, it is a compact
;; representation of a set of possible traces of our model, suppressing
;; information in the intermediate nodes.
