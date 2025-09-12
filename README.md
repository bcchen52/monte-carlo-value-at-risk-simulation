# monte-carlo-value-at-risk-simulation

A Monte Carlo-based simulation tool for visualizing portfolio risk, expected returns, volatility, and using that to determine value at risk.

Given a portfolio (tickers and values), and some simulation parameters (time, confidence interval, number of simulations), we return the VaR with a specified confidence.

Using the specific tickers' values from the last year, we find the mean and covariance matrix to estimate parameters for the geometric brownian motion equation, which is put in a monte carlo simulation for x times.

Below is the Stochastic Differential Equation (SDE) that defines Geometric Brownian Motion (GBM)...

$$\dfrac{dS_t}{S_t} = \mu dt + \sigma dW_t$$

The equation used for simulation is altered and discretized...

$$S_t = S_{t-\Delta t} \cdot e^{\Delta t(\mu - \dfrac{1}{2} \sigma^2) + \sigma \sqrt{\Delta t} \cdot Z}, Z \sim \mathcal{N} (0, 1)$$

Or, simply...

$$S_t= S_{t-\Delta t} \cdot \exp(\Delta t(\mu - \dfrac{1}{2} \sigma^2) + \sigma \sqrt{\Delta t} \cdot Z), Z \sim \mathcal{N} (0, 1).$$

This asset-price model is used in the Black-Scholes option-pricing formula

# Table of Contents
- [Future Changes](#future-changes)
- [Mathematical Explanation](#mathematical-explanation)
- [Functionality](#functionality)

# Future Changes
1. Finish README and sources
2. Change CVaR calculation to remove negative values
3. Go beyond basic GBM, fat tails, etc...

# Mathematical Explanation

Though GBM has been the foundation of decades-old financial models, exploring the stochastic calculus and statistics used to solve the GBM SDE was pretty interesting. 

The defining SDE for Geometric Brownian Motion

$$\dfrac{dS_t}{S_t} = \mu dt + \sigma dW_t,\tag{1}$$

can be solved into a form easy to model, 

$$S_t = S_{t-1} \cdot e^{\Delta t(\mu - \dfrac{1}{2} \sigma^2) + \sigma \sqrt{\Delta t} \cdot Z}, Z \sim \mathcal{N} (0, 1). \tag{2}$$

This solution is found from using $\log_e(S_t)$, the natural logarithm, though for the remainder of this explanation I will use $\log(x)$ to denote the natural logarithm $\ln(x)$. Specifically, we aim to solve

$$d\log(S_t),$$

where

$$dS_t = S_t\mu dt + S_t\sigma dW_t. \tag{3}$$

First, we need to introduce a Wiener process or Brownian motion, $W_t$, which, summarized from Wikipedia, follows that $^{[1]}$

1. $W$ is an almost surely continuous martingale with $W_0 = 0$
2. $W$ has independent increments: $\forall t > 0$, increments $W_{t+u} - W_t$, $u \geq 0$, are independent of $W_s$, $s<t$
3. $W$ has Gaussian increments: $W_{t+u} - W_t \sim \mathcal{N} (0,u)$, where variance is the length of the interval, $u$
4. $W$ has almost surely continuous paths, and is not differentiable anywhere.

We apply a logarithm to $(1)$ because the SDE is not easily solvable with direct integration because $dW_t$, and thus $\int{dW_t}$, does not exist due to the non-differentiability of a Wiener process, mentioned as the 4th property above. Additionally, logarithms have some important contributions to financial modeling, [more on logarithms here](#more-on-logarithms).

## Itô's Lemma for an Itô Process

Itô's Lemma states that for a function $f(X_t, t)$ where $X_t$ is an Itô process that satisfies the stochastic differential equation...

$$dX_t = \mu_t dt + \sigma_t dW_t, \tag{4}$$

where $B_t$ is a Wiener process/Brownian motion 

$$df(X_t, t) = \dfrac{\partial{f}}{\partial{t}}dt + \mu_t \dfrac{\partial{f}}{\partial{X_t}}dt + \dfrac{1}{2}\dfrac{\partial^2{f}}{\partial{X_t^2}}\sigma_t^2dt + \dfrac{\partial{f}}{\partial{X_t}}\sigma_t dW_t. \tag{5}$$
$^{[2]}$

Or, equivalently in a more general form

$$df(X_t, t) = \dfrac{\partial{f}}{\partial{t}}dt + \dfrac{\partial{f}}{\partial{X_t}}dX_t  + \dfrac{1}{2}\dfrac{\partial^2{f}}{\partial{X_t^2}}\sigma_t^2dt$$

where $f=u$ and $\sigma_t^2=v(X_t)$.$^{[3]}$

Solving the terms in $(5)$ for $d\log(S_t)$ we have 

$${\dfrac {\partial \log(S_t)}{\partial S_t}} = \dfrac{1}{S_t} \tag{6}$$

$${\dfrac {\partial^2 \log(S_t)}{\partial S_t^{2}}} = -\dfrac{1}{S_t^2}, \tag{7}$$

from the derivative of $\log$ and

$$\dfrac{\partial{\log(S_t)}}{\partial{t}} = 0$$

because $S_t$ is not dependent on time, giving us: 

$$d(\log(S_t)) = (\dfrac{1}{S_t}\mu_t - \dfrac{1}{2}\dfrac{1}{S_t^2}\sigma_t^2)dt + \dfrac{1}{S_t}\sigma_t dW_t.$$

Note that this is defined for an Itô process defined by the SDE $(4)$, and using the equivalent form of the GBM SDE $(3)$ we have 

$$\mu_t = S_t \mu$$

$$\sigma_t = S_t \sigma$$

giving us:

$$d\log(S_t) = (\mu - \dfrac{1}{2}\sigma^2)dt + \sigma dW_t. \tag{8}$$

Below is an explanation of the motivation for Itô's lemma and how we use calculus to arrive at the same conclusion, as well as a deeper dive into some stochastic concepts. Or, skip directly to the next part, [discretizing dlog(S_t)](#discretizing-solved-differential).

### Itô's Lemma Intuition
---
[Goodman](https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson4.pdf)$^{[3]}$ offers a rigorous proof for Ito's lemma using convergence, integrals, and expected value. 

Using the chain rule for some $f=g(h(x))$ requires $h$ to be differentiable at $x$ and $g$ differentiable at $h(x)$. The chain rule uses the limit definition of $\frac{df}{dt}$, and uses the limit definition of the derivative of $h$ within that definition — however, we know that a Wiener process $W_t$ is nowhere differentiable from its fourth property and our $h(x) = S_t $ involves $W_t$ (1).

First, we look at taylor expansion of $f$...

$$f(x) = f(a) + f'(a)(x-a) + \dfrac{f''(a)}{2!}(x-a)^2$$

$$+\space \dfrac{f'''(a)}{3!}(x-a)^3 + \dots$$

$^{[4]}$

We define $\Delta f = f(x) - f(a)$ and $x-a$ as $\Delta x$, thus

$$\Delta f = f'(a)\Delta x + \dfrac{f''(a)}{2}(\Delta x)^2 + \dfrac{f'''(a)}{6}(\Delta x)^3 + \dots$$

Then, we look at differentials, i.e. infinitesimally small changes:

$$df = f'dx + \dfrac{f''}{2}(dx)^2 + \dfrac{f'''}{6}(dx)^3 + \dots$$

In our case, we have $f(S_t,t)$ and need to look at the derivatives as total derivatives w.r.t. partial derivatives up to the second order term:

$$df = \dfrac{\partial f}{\partial S_t}dS_t + \dfrac{\partial f}{\partial t}dt + \dfrac{1}{2}\left(\dfrac{\partial^2 f}{\partial S_t^2}(dS_t)^2 + \dfrac{\partial^2 f}{\partial S_t \partial t}dS_tdt  + \dfrac{\partial^2 f}{\partial t^2}(dt)^2\right)$$

We only decompose up to the second term, which will be explained later. 

Using $f(S_t,t) = \log(S_t)$ we have:

$$\dfrac{\partial \log(S_t)}{\partial S_t} = \dfrac{1}{S_t}$$

$$\dfrac{\partial^2 \log(S_t)}{\partial S_t^2} = -\dfrac{1}{S_t^2}$$

and anything that involves some $f(S_t)$ and $\partial t$ of any order is $0$, because $S_t$ does not explicitly depend on $t$ $(1)$.

Thus,

$$d\log(S_t) = \dfrac{dS_t}{S_t} - \dfrac{1}{2}\dfrac{(dS_t)^2}{S_t^2}.$$

We know that $\dfrac{(dS_t)^2}{S_t^2} = \left(\dfrac{dS_t}{S_t}\right)^2$, and $\dfrac{dS_t}{S_t}$ is defined in $(1)$, giving us:

$$d\log(S_t) = (\mu dt + \sigma dW_t) + \dfrac{1}{2}(\mu dt + \sigma dW_t)^2$$

$$= (\mu dt + \sigma dW_t) + \dfrac{1}{2}\left(\mu^2 (dt)^2 + 2 \mu \sigma dW_tdt + \sigma^2 dW_t^2\right).$$

The next step requires the use of Itô's multiplication rules, which state that for a wiener process $W_t$... [here](https://www.uni-rostock.de/storages/uni-rostock/Alle_MNF/Mathematik/Struktur/Lehrstuehle/Angewandte_Analysis/Erasmus_Program/12-06-Pospisil-Fin.pdf?)
$$dW_t^2 = dt \tag{9}$$
$$dW_t \cdot dt = 0 \tag{10}$$
$$dt \cdot dt = 0 \tag{11}$$
$^{[5]}$

Applying these rules, we can reduce to:

$$d\log(S_t) = \mu dt + \sigma dW_t + \dfrac{1}{2}\sigma^2dt$$

$$= (\mu - \dfrac{1}{2}\sigma^2)dt + \sigma dW_t,$$

the same as we found when applying Itô's lemma directly $(5)$.

To understand how we arrive at these rules, read the following section. Or, skip to [discretizing $d\log(S_t)$](#discretizing-solved-differential) to see the next steps.

#### Understanding Itô's Multiplication Rules
---
$(9)$ is the most interesting to prove and can be understood through the statistical properties of a Wiener process. 

To determine $dW_t^2 = dt$, the idea is to show some $\int dW_t^2 = t$ using a limit definition of an integral, such as the Riemann–Stieltjes integral that we can evaluate using Wiener process properties:

$$\int dW_t^2 = \lim\limits_{n \rightarrow \infty} \sum\limits_{i=1}^n (W_{i} - W_{i- \Delta t})^2.$$

However, this integral does not exist because $W_t$ is not differentiable so $dW_t$ and thus $dW_t^2$ does not exist. The Itô stochastic integral

$$\int_0^t H_sdX_s = \lim\limits_{n \rightarrow \infty}\sum\limits_{[t_{i-1}, t_i] \in \pi_n}H_{t_{i-1}}(X_{t_i} - X_{t_{i-1}})$$

gives meaning to $dX_s$, where $X_s$ is a semimartingale, with a definition similar to the previous integral. Note that we defined a Wiener process $W$ as a martingale earlier, and thus, it is a semimartingale. However, this does not apply to $dX_s^2$.

Instead, we can use the *quadratic covariation* of semimartingales,

$$[X, Y]_t = \lim\limits_{n \rightarrow \infty}\sum_{i=1}^n(X_t-X_{t-i})(Y_t-Y_{t-i}), \tag{12}$$

which follows the property 

$$\Delta[X, Y]_t = \Delta X \Delta Y. \tag{13}$$

$^{[5]}$

More specifically, the *quadratic variation* is the quadratic covariation of the same process, 

$$[X]_t = [X,X]_t.$$

Then, we discretize the changes in $(13)$ as derivatives and apply to quadratic variation, giving us

$$d[X]_t = dX^2,$$

where we can solve for $[X]_t$.

There are many ways to prove $[W] = t$, including with more stochastic calculus, but evaluating $(12)$ directly is the approach most intuitive to me. 

We solve

$$[W]_t = \lim\limits_{n \rightarrow \infty} \sum\limits_{i=1}^n (W_{i} - W_{i- \Delta t})^2,$$

where $n \Delta t = t$ and $W_{i} - W_{i- \Delta t}$ is an increment of a Wiener process. Wiener process properties 2 and 3 tell us these increments are independent and $W_{i} - W_{i- \Delta t} \sim \mathcal{N}(0, \Delta t)$. Note the similarity to the Riemann–Stieltjes integral definition.

Since $W_t$ is a random variable, we look at expected value and solve for 

$$\mathbb{E}\left[\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2\right].$$

First, we find $\mathbb{E}[(W_{i} - W_{i- \Delta t})^2]$ by solving for the second moment $\mathbb{E}[X^2]$ using variance

$$\mathrm{Var}[X] = \mathbb{E} \left[(X - \mathbb{E}[X])^2\right] \tag{14}$$

$$= \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \tag{15}$$

with $X = W_{i} - W_{i- \Delta t}.$ This gives us 

$$\mathrm{Var}[W_{i} - W_{i- \Delta t}] = \Delta t$$

and 

$$\mathbb{E}[W_{i} - W_{i- \Delta t}]^2 = 0$$

from the Gaussian increment property. Thus

$$\mathbb{E}[(W_{i} - W_{i- \Delta t})^2] = \Delta \tag{16}t$$

and

$$\mathbb{E}[\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2] = \sum\limits_{i=1}^n\mathbb{E}[(W_{i} - W_{i- \Delta t})^2] = \sum\limits_{i=1}^n \Delta t = n\Delta t = t, \tag{17}$$

using the linearity of expectation because increments of $W$ are independent. Note earlier how I defined $n \Delta t = t$, allowing $\Delta t$ to become infinitesimally small as $n \rightarrow \infty$.

Expected value alone does not give us enough information to assign a value to our random variables — we need to look at variance. 

To find $\mathrm{Var}[\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2]$, we use $(15)$ with $X = (W_{i} - W_{i- \Delta t})^2$ and solve

$$\mathrm{Var}[(W_{i} - W_{i- \Delta t})^2] = \mathbb{E}[(W_{i} - W_{i- \Delta t})^4] - (\mathbb{E}[(W_{i} - W_{i- \Delta t})^2])^2.$$

We already calculated the last term, the second moment, in $(16)$, so

$$(\mathbb{E}[(W_{i} - W_{i- \Delta t})^2])^2 = \Delta t^2.$$

The earlier term is the 4th moment. From standard probability theory, the $nth$ moment $\mathbb{E}[X^n]$ is defined as the $nth$ derivative of the moment generating function (MGF) at $t=0$, or 

$$\mathbb{E}[X^n] = M_X^n(0).$$

The MGF for a normal random variable is derived with its CDF, $f_X(x)$, which I omit, giving us

$$M_X (t) = \mathbb{E}[e^{tX}] = \int_{-\infty}^\infty e^{tx} f_X(x) dx = ... = \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right). $$

We could use this approach for the second moment, but using the variance formula is easier. Taking the derivatives is lengthy, but repeatedly differentiating $\exp$ allows us to combine a few terms...

$$M_X^1 (t) = (\mu + \sigma^2 t)\exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$

$$M_X^2 (t) = (\mu + \sigma^2 t)^2 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)+\sigma^2 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$

$$M_X^3 (t) = (\mu + \sigma^2 t)^3 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right) + 2\sigma^2(\mu + \sigma^2 t)\exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$
$$+\space \sigma^2(\mu + \sigma^2 t)\exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$

$$= (\mu + \sigma^2 t)^3 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right) + 3\sigma^2(\mu + \sigma^2 t)\exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$

$$M_X^4 (t) = (\mu + \sigma^2 t)^4 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)+ 3\sigma^2(\mu + \sigma^2 t)^2 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)$$

$$ + \space 3\sigma^2\left((\mu + \sigma^2 t)^2 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)+\sigma^2 \exp\left(\mu t + \dfrac{1}{2} \sigma^2 t^2 \right)\right).$$

Then, solving for $t=0$ gives us a nicer equation...

$$M_X^4 (0) = \mu^4 + 3\sigma^2\mu^2 + 3\sigma^2(\sigma^2+\mu^2)$$

$$= \mu^4 + 6\sigma^2\mu^2 + 3\sigma^4.$$

Further simplifying with $\mu = 0$, we have

$$\mathbb{E}[(W_{i} - W_{i- \Delta t})^4] = 3\sigma^4$$

and thus 

$$\mathrm{Var}[(W_{i} - W_{i- \Delta t})^2] = 3(\Delta t)^2 - \Delta t^2 = 2\Delta t^2 = 2(\dfrac{t}{n})^2.$$

Finally, we have

$$\mathrm{Var}[\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2] = \sum\limits_{i=1}^n\mathrm{Var}[(W_{i} - W_{i- \Delta t})^2]$$

$$= \sum\limits_{i=1}^n 2(\dfrac{t}{n})^2 = 2\dfrac{t^2}{n}, \tag{18}$$

again because the increments are independent. 

Note that

$$2\dfrac{t^2}{n} \xrightarrow{n \rightarrow \infty} 0. \tag{19}$$

This behavior is especially important when we consider that we were originally looking at the limit as $n \rightarrow \infty$, which in our case represents an infinitesimally small increment.

Intuitively, it's reasonable to expect

$$\lim\limits_{n \rightarrow \infty} \sum\limits_{i=1}^n (W_{i} - W_{i- \Delta t})^2 = t$$

given $(17)$ and $(19)$.

We can formally claim that $[W] = t$ by proving that $\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2$ converges to $t$ in mean square (also known as quadratic mean), which is stronger than convergence in probability.$^{[6]}$ A sequence of random variables $X_n$ converges in mean-square to a random variable $X$, $X_n \xrightarrow{m.s.} X$, if...

$$\lim\limits_{n \rightarrow \infty} \mathbb{E}\left[(X_n - X)^2\right] = 0.$$

We define $X=t$ and $X_n = \sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2$ and need to show 

$$\lim\limits_{n \rightarrow \infty} \mathbb{E}\left[(X_n - t)^2\right] = 0.$$

From $(17)$ we know that $\mathbb{E}\left[X_n\right] = t$, and we have

$$\mathbb{E}\left[(X_n - t)^2\right] = \mathbb{E}\left[(X_n - \mathbb{E}\left[X_n\right])^2\right] = \mathrm{Var}[X_n],$$

$^{[7]}$ the original definition of variance $(14)$. 

We solved for variance in $(18)$, and we already observed from $(19)$ that

$$\lim\limits_{n \rightarrow \infty} 2\dfrac{t^2}{n} = 0,$$

proving $\sum\limits_{i=1}^n(W_{i} - W_{i- \Delta t})^2 \xrightarrow{m.s.} t.$

Thus 

$$[W]_t = \lim\limits_{n \rightarrow \infty} \sum\limits_{i=1}^n (W_{i} - W_{i- \Delta t})^2 = t$$

and 

$$d[W]_t = dW_t^2 = dt.$$

There are similar proofs for $(10)$ and $(11)$, but they can be simply understood in terms of the scale of $dt$, as $dt$ is an infinitesimally small increment of time. 

For $(10)$ we have

$$dWt \cdot dt = \sqrt{dt} \cdot dt = dt^{3/2},$$

and for $(11)$ we have 

$$dt \cdot dt = dt^2,$$

which are both a higher order of magnitude than $dt$, and when $dt$ is small, $dt^2$ and $dt^{3/2}$ are even smaller and are thus negligable.

## Discretizing solved differential

Now that we've solved for

$$d\log(S_t) = (\mu - \dfrac{1}{2}\sigma^2)dt + \sigma dW_t,$$

we want to discretize into something that we can incrementally model. 

We treat each derivative as an increment of time

$$\Delta\log(S_t) = (\mu - \dfrac{1}{2}\sigma^2)\Delta t + \sigma \Delta W_t,$$

and further turn these into increments 

$$\log(S_t) - \log(S_{t-\Delta t}) = (\mu - \dfrac{1}{2}\sigma^2)\Delta t + \sigma  (W_t - W_{t-\Delta t}). \tag{20}$$

We move $log(S_{t-\Delta t})$ and take the exponential...

$$S_t = S_{t-\Delta t} \cdot \exp \left( (\mu - \dfrac{1}{2}\sigma^2)\Delta t + \sigma (W_t - W_{t-\Delta t}) \right).$$

Then, using what we know about increments of $W$, we have

$$W_t - W_{t-\Delta t} \sim \mathcal{N}(0, \Delta t).$$

However, we can further simplify a standard normal random variable by using the linearity of the normal distribution. More specifically, given a normal random variable $X \sim \mathcal{N} (\mu,\sigma^2)$ and another random variable $Y=bX$ where $b$ is a constant...

$$\mathbb{E}[Y] = b\mathbb{E}[X] \tag{21}$$

$$\mathrm{Var}(Y) = b^2\mathrm{Var}(X). \tag{22}$$

$^{[8]}$

As a refresher, for a random normal variable $X \sim \mathcal{N} (\mu,\sigma^2)$, $\mathbb{E}[X] = \mu$ and $\mathrm{Var}(Y) = \sigma^2$. Comparing $W_t - W_{t-\Delta t}$ to $Z \sim \mathcal{N} (0,1)$,

$$\mathrm{Var}(W_t - W_{t-\Delta t}) = \Delta t = (\sqrt{\Delta t})^2 \cdot 1 = (\sqrt{\Delta t})^2 \cdot \mathrm{Var}(Z),$$

satisfying $(21).$ To confirm that $b = \sqrt{\Delta t},$

$$\mathbb{E}[W_t - W_{t-\Delta t}] = 0 = \sqrt{\Delta t} \cdot 0 = \sqrt{\Delta t} \cdot \mathbb{E}[Z],$$

satisfying $(22)$. Thus,

$$W_t - W_{t-\Delta t} \sim \sqrt{\Delta t}\cdot Z.$$

Finally, the final equation used in simulation...
$$S_t= S_{t-\Delta t} \cdot \exp(\Delta t(\mu - \dfrac{1}{2} \sigma^2) + \sigma \sqrt{\Delta t} \cdot Z), Z \sim \mathcal{N} (0, 1).$$

## Parameter Estimation

We have to estimate the parameters $\mu$ and $\sigma$, which are by definition the instanteous drift and volatility of $S_t$, price. One might initially consider the sample mean and standard deviation of raw returns over a time period to estimate these parameters, but doing so is not the most straightforward approach. 
- Consider that raw returns are cumulative, while log returns are additive, and if we wanted to discretize the instantenuous SDE into a time period via the Euler-Maryama method by combining a range of individual time steps we would have returns equal to a product of $\mu$

Luckily, we have the result from our solved SDE (INSERT) that log returns define an brownian motion with drift and volatilaity and thus 

$$\ln(\dfrac{S_t}{S_{t-\Delta t}})\sim \mathcal{N}((\mu - \dfrac{1}{2} \sigma^2)\Delta t, \sigma^2 \Delta t).$$

Denote $X_t$ as the logarithm return at time $t$ over a period of $\Delta t$,

$$X_t = \ln(\dfrac{S_t}{S_{t-\Delta t}}).$$

From the normal distribution,

$$\mathbb{E}[X_t] = (\mu - \dfrac{1}{2} \sigma^2)\Delta t = \mu_{ln},$$

$$\mathrm{Var}(X_t) = \sigma^2 \Delta t.$$

From statistics, the general unbiased and consistent estimators for the mean and variance of an i.i.d. sequence of random variables are the sample mean and the unbiased sample variance, 

$$\hat{\mu_{ln}}=\bar{X_n}=\dfrac{1}{n}\sum^{n}_{i=1}{X_i},$$

$$\hat{\sigma^2}=s^2=\dfrac{\sum^n_{i=1}{(X_i-X_n)^2}}{n-1}.$$

Since we know the distribution of $X_t$, we can also use MLE to find estimators, resulting in sample mean and the biased sample variance.

Additivity of logarithms allows us to simplify calculation of the sum in the sample mean,

$$\sum_{i=1}^n\log(\dfrac{S_i}{S_{i-1}})=\sum_{i=1}^n\log(S_i)-\log({S_{i-1}})$$

$$=\log(S_n)-\log(S_0),$$

because it is a telescoping sum where all terms but $n$ and $0$ cancel.

However, we still need to calculate log returns at every point for sample variance. 

## Covariance and Correlation

Work in progress...

Sources to be cited...

https://www.math.uchicago.edu/~lawler/finbook.pdf gives us definition of Brownian motion (all properties, arithmetic), quadratic variation, and then introduces multidimensional with covariance matrix. Also Ito's formula and derivation.

https://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-BM-GBM-I.pdf multivariate gbm and covariance deifnitions as well as arithmetic BM clear statement. Basis for covariance.
https://www.columbia.edu/~ks20/4703-07/4703-08-CVN-syllabus.html is the full course.

https://www.mathematik.uni-muenchen.de/~philip/publications/lectureNotes/philipPeter_NumMethodsForMathFin.pdf More on multidimensional GBM.

https://arxiv.org/pdf/0812.4210 next steps

## More on Logarithms

Logarithms, specifically *log returns*, defined as 

$$\log(r_t+1) = \log(\frac{p_t}{p_{t-1}}), \tag{23}$$

where 

$$r_t=\frac{p_t - p_{t-1}}{p_{t-1}}=\frac{p_t}{p_{t-1}}-1$$

is return from time $t-1$ to $t$ and $p_t$ is price at time $t$, are very useful for financial modeling due to their statistical properties and ease of computation.

In the equation we use to model price $(2)$, we incorporate log returns. This can be seen when transforming $(20)$,

$$\log\left(\dfrac{S_t}{S_{t-\Delta t}}\right) = (\mu - \dfrac{1}{2}\sigma^2)\Delta t + \sigma  (W_t - W_{t-\Delta t}), \tag{24}$$

which matches $(23)$. 

Notice that GBM defines log returns as a function of a normal random variable (increment of $W$), thus making log returns follow a normal distribution. Historical data generally follows this pattern, and is a large part of why GBM has been used for financial models. 

Log returns following a normal distribution is also beneficial for modeling price. A random variable $X$ is *log-normally distributed* if 

$$log(X) \sim \mathcal{N} (\mu, \sigma^2).$$

Looking directly at our final equation for $S_t$ $(2)$ or distributing the denominator in $(24)$, we can see that 

$$log(S_t) \sim \mathcal{N} (\mu_s, \sigma_s^2)$$

and thus $S_t$, price, is log-normal. This allows price to not be negative due to the nature of logarithms — logarithms are undefined for real negative values and conversely the exponential of a real number cannot be negative. 

Gunderson says more about the use of log prices and returns, including some more statistical properties, on his [blog](https://gregorygundersen.com/blog/2022/02/06/log-returns/).

Another interesting advantage of logarithms is numerical stability of floating point from a computational standpoint. Logarithms are additive over multiplication,

$$log(xy) = log(x) + log(y),$$

allowing us to find and model compounded returns with addition rather than multiplication. 

Both addition of log returns and multiplication of raw returns face rounding errors when modeled with floating point, but error in multiplication accumulates exponentially compared to linearly in addition, which makes multiplication especially worse over many inputs. 

Though both operation can have catastrophic error with edge cases, we look at the more realistic case of lost precision.
- When adding with logarithms, we lose precision because logarithms themselves are estimates with limited precision
- In terms of precision lost from the operations themselves, multiplication ___
- We can look at an example of this in decimal as a heuristic — take the example of 1.05 and 0.983, which are in the range of realistic raw changes. When we multiply then, we have $1.05 \cdot 0.983 = 1.03215$, where the number of the product's decimal bits is the sum of the factors'. 

This does not directly apply to floating point, since fractions are handled differently. For example, the mantissa for 1.05 is XXX, which almost takes up the entire precision available, but the point remains that multiplication requires amplifies precision, as it does with error, so you sort of have a double 

## References

[1] Rick Durrett. *Probability: Theory and Examples*, Version 5. January 11, 2019. Available at: https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf

[2] Younggeun Yoo. *Stochastic Calculus and Black-Scholes Model*. REU Research Paper, University of Chicago, 2017. Available at: https://math.uchicago.edu/~may/REU2017/REUPapers/Yoo.pdf

[3] Jonathan Goodman. *Stochastic Calculus*. Courant Institute, Fall 2018. Lesson 4, Ito's Lemma. Available at: https://math.nyu.edu/~goodman/teaching/StochCalc2018/notes/Lesson4.pdf

[4] Elias Zakon. *Mathematical Analysis*. University of Windsor via the Trilla Group. Section 5.6: Differentials, Taylor's Theorem and Taylor's Series. Available at: https://math.libretexts.org/Bookshelves/Analysis/Mathematical_Analysis_(Zakon)/05%3A_Differentiation_and_Antidifferentiation/5.06%3A_Differentials._Taylors_Theorem_and_Taylors_Series#:~:text=%CE%94f%3Ddf%2Bo,within%20o(%CE%94x).&text=where%20f(n)%20is%20the,(at%20p%20and%20x)

[5] Andreas Eberle. *Stochastic Analysis*. January 28, 2013. Available at: https://wt.iam.uni-bonn.de/fileadmin/WT/Inhalt/people/Andreas_Eberle/StoAn1213/StochasticAnalysisNotes1213.pdf

[6] Larry Wasserman. *Chapter 5: Convergence of Random Variables*. Carnegie Mellon University, Department of Statistics. Available at: https://www.stat.cmu.edu/~larry/%3Dstat325.01/chapter5.pdf

[7] Marco Taboga. *Mean-Square Convergence*. StatLect, Asymptotic Theory. Available at: https://www.statlect.com/asymptotic-theory/mean-square-convergence

[8] Marco Taboga. *Linear Combinations of Normal Random Variables*. StatLect, Probability Distributions. Available at: https://www.statlect.com/probability-distributions/normal-distribution-linear-combinations

# Functionality

Work in progress, explanation of each visualization and how.





