# stats_test_from_scratch
One issue with Python is that there is no unified source for statistical tests like there is for ML with scikit-learn or Deep Learning with Keras. 

You could have statistical tests in Scipy, Statsmodels, Numpy and Pandas; but it's not clear which libraries carry which tests. Additionally, there are some tests that you can find in R that aren't currently supported by these libraries.

My goal is two fold:
1) Implement the statistical tests from scratch, with an emphasis on making the code as presentable and easy-to-understand as possible. This is so anyone can understand exactly how what the test is measuring and how it is measuring it, even if it means we are sacrificing computational speed in the process.
2) Identify where in Python you can find these test(s), if at all. That way, for those who want the fastest implementation, they'll understand where to find the test.

# Statistical Tests currently supported and where to find them:
## Sample Tests
1) One and two sample Z Tests: Statsmodels through [ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html). Used to determine if the sample differs significantly from the normally distributed population we are evaluating, or if the distribution of two samples from a normally distributed population differ. 
2) One and two sample T Tests: Scipy through [ttest_1samp](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_1samp.html) and [ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html). Used to determine if the sample differs significantly from the normally distributed population (with unknown sample variance), or if the means of two samples from a normally distributed population differ. 
3) Trimmed Means T Test: Not found in either scipy or statsmodels. Used when our two samples violate the assumption of normality.
4) Yeun-Welch Test: Not found in either scipy or statsmodels. Used when our two samples violate the assumption of normality and equality of variances.
5) Two Sample F Test: Not found in either scipy or statsmodels. Used to determine if the variances of two populations are equal. 
6) Binomial Sign Test: Statsmodels through [sign_test](https://www.statsmodels.org/stable/generated/statsmodels.stats.descriptivestats.sign_test.html#statsmodels.stats.descriptivestats.sign_test). Used to determine if there are consistent significant differences between pairs of data, such as before-and-after treatments.
7) Wald-Wolfowitz Test: Statsmodels through [runstest_1samp](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.stats.runs.runstest_1samp.html#statsmodels.sandbox.stats.runs.runstest_1samp). Used to determine if the elements of a dataset are mutually independent.
8) Trinomial Test: Not found in either scipy or statsmodels. Used as a replacement to the sign test when there are ties in the data.

## Rank Tests
1) Wilcoxon Rank-Sum Test: Scipy through [wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html). Used to determine if two related or paired samples have different mean ranks. 
2) Mann-Whitney-U Test: Scipy through [mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html). Used to determine if a randomly selected value from one ordinal population will be less or greater than a randomly selected value from a second ordinal population.
3) Friedman Test: Scipy through [friedmanchisquare](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html). Used to determine if there are any differences in treatments across multiple test attempts.
4) Quade Test: Not found in either scipy or statsmodels. Used to determine if there is at least one treatment that is different from the others.
5) Page's Trend Test: Not found in either scipy or statsmodels. Used to determine if the central tendency for all treatments is the same, or there is an order to them.
6) Kruskal-Wallis Test: Scipy through [kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html). Used to determine if two or more samples originate from the same distribution.
7) Fligner-Kileen Test: Scipy through [fligner](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fligner.html). Used to determine if two or more samples have the same variances without the assumption of normality.
8) Ansari-Bradley Test: Scipy through [ansari](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ansari.html). Used to determine if two samples have the same dispersion (distance from the median).
9) Mood Test for Dispersion: Scipy through [mood](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mood.html). Used to determine if two samples have the same dispersion for their ranks.
10) Cucconi Test: Not found in either scipy or statsmodels. Used to determine if the central tendency and variability of two samples are the same.
11) Lepage Test: Not found in either scipy or statsmodels. Used to determine if the central tendency and variability of two samples are the same.
12) Conover Test: Not found in either scipy or statsmodels. Used to determine if the variances of multiple groups are the same. 

## Categorical Tests
1) Chi Square Test: Scipy through [chi2_contingency](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html). Used to determine if the distribution of our contingency table follows the row and column sum.
2) G Test: Scipy through [chi2_contingency(lambda_="log-likelihood")](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html). Used to determine the likelihood that our contingency follows the distribution of our row and column sum. 
3) Fisher Test: Scipy through [fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html). Used to determine the exact likelihood that we would observe a measurement that is more extreme than our expected results.
4) McNemar Test: Statsmodels through [mcnemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html). Used to determine if the marginal row and column probabilities are equal.
5) Cochran–Mantel–Haenszel Test: Statsmodels through [StratifiedTable.test_null_odds](https://www.statsmodels.org/dev/generated/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds.html#statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds). Used to determine if there is an association between a binary predictor/treatment and a binary outcome across all strata. 
6) Woolf Test: Not found in either scipy or statsmodels. Used to determine if there exists the same log odds across all strata.
7) Breslow-Day Test: Found in statsmodels as [StratifiedTable.test_equal_odds()](https://www.statsmodels.org/dev/generated/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_equal_odds.html#statsmodels.stats.contingency_tables.StratifiedTable.test_equal_odds). Used to determine if there exists the same odds ratio across all strata.
8) Bowker Test: Found in statsmodels as [TableSymmetry or as bowker_symmetry](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.SquareTable.symmetry.html#statsmodels.stats.contingency_tables.SquareTable.symmetry). Used to determine if the proportions between two treatments are symmetrical. 

## Multi-Group Tests
1) Levene Test: Scipy through [levene(center='mean')](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html). Used to determine the equality of group variances using the distance from the mean.
2) Brown-Forsythe Test: Scipy through [levene(center='median')](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html). Used to determine the equality of group variances using the distance from the median.
3) One Way F-Test: Scipy through [f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html). Used to determine the equality of group means.
4) Bartlett Test: Scipy through [bartlett](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html). Used to determine the equality of group variances using the likelihood ratio. 
5) Tukey Range Test: Statsmodels through [pairwise_tukeyhsd](https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html). Used to determine the equality of means for all sample pairs.
6) Cochran's Q Test: Statsmodels through [cochrans_q](https://www.statsmodels.org/devel/generated/statsmodels.stats.contingency_tables.cochrans_q.html). Used to determine if the treatments (as measured by a binary response variable) have identical effects/are equally effective.
7) Jonckheere Trend Test: Not found in either scipy or statsmodels. Used to determine if the group medians have an a-priori ordering.
8) Mood Median Test: Scipy through [median_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.median_test.html). Used to test the equality of group medians.
9) Dunnett Test: Not found in either scipy or statsmodels. Used as post-hoc to ANOVA analysis to determine which groups are significantly different to the control group.
10) Duncan's New Multi-Range Test: Not found in either scipy or statsmodels. Used as post-hoc to ANOVA analysis to determine which group means are significantly different to one another.

## Proportion Tests
1) One and two sample Proportion Z Tests: Statsmodels through  [proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html). Used to determine if one proportion is different to the population proportion mean, or if two proportions share the same mean. 
2) Binomial Test: Scipy through [binom_test](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom_test.html). Used to determine if the sample follows a given binomial distribution. 
3) Chi Square Proportion Test: Not found in either scipy or statsmodels. Used to determine if the proportion within groups follows a population distribution.
4) G Proportion Test: Not found in either scipy or statsmodels. Used to determine if the distribution of groups follows a population distribution.

## Goodness of Fit Tests
1) Shapiro-Wilk Test: Scipy through [shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html). Used to determine if a random sample is derived from a normal distribution.
2) Chi Goodness of Fit Test: Scipy through [chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html). Used to determine if the distribution of groups follows an expected result.
3) G Goodness of Fit Test: Scipy through [power_divergence(lambda_="log-likelihood")](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.power_divergence.html). Used to determine if the distribution of groups follows an expected result.
4) Jarque-Bera Test: Statsmodels through [jarque_bera](https://www.statsmodels.org/devel/generated/statsmodels.stats.stattools.jarque_bera.html). Used to determine if the sample's skew and kurtosis follow that of a normal distribution.
5) Ljung-Box Test: Statsmodels through [acorr_ljung(boxpierce=False)](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html). Used to determine if the autocorrelations are equal to 0.
6) Box-Pierce Test: Statsmodels through [acorr_ljung(boxpierce=True)](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html). Used to determine if the autocorrelations are equal to 0.
7) Skew Test: Scipy through [skewtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html). Used to determine if the sample is normally distributed through its skew.
8) Kurtosis Test: Scipy through [kurtosistest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html). Used to determine if a sample is normally distributed through its kurtosis.
9) K-Squared Test: Scipy through [normaltest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html). Used to determine if a sample is normally distributed through its skew and kurtosis. 

## Correlation Tests
1) Pearson Test: Scipy through [pearsonr](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pearsonr.html). Used to determine the correlation between two different data points.
2) Spearman Rank Test: Scipy through [spearmanr](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html). Used to determine the correlation between the ranks of two different data points.
3) Kendall-Tau Test: Scipy through [kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html). Used to determine the correlation between two ordinal variables.
4) Point Biserial Correlation: Scipy through [pointbiserialr](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.pointbiserialr.html). Used to determine the correlation between two variables when one of them is dichotomous.
5) Rank Biserial Correlation: Not found in either scipy or statsmodels. Used to determine the correlation between two variables when one of them is dichotomous and the other consists of ranks.

## Outliers Tests
1) Tukey's Fence Test: Not found in either scipy or statsmodels. Used to determine outliers based on their distance from the first or third quartile. 
2) Grubb's Test: Not found in either scipy or statsmodels. Used to determine if there exists one outlier in the dataset.
3) Extreme Studentized Deviant (ESD) Test: Not found in either scipy or statsmodels. Used to determine if there exists up to k outliers in the dataset, with k specified by the user. 
4) Tietjen-Moore Test: Not found in either scipy or statsmodels. Used to determine if there exists k outliers in the dataset, with k specified by the user.
5) Chauvenet Test: Not found in either scipy or statsmodels. Used to determine outliers based on the Chauvenet criteria.
6) Peirce Test: Not found in either scipy or statsmodels. Used to determine outliers based off of Peirce's criteria.
7) Dixon's Q Test: Not found in either scipy or statsmodels. Used to determine outliers based on the Q values.
8) Thompson-Tau Test: Not found in either scipy or statsmodels. Used to determine outliers based on the Thompson-Tau criteria.
9) MAD-Median Test: Not found in either scipy or statsmodels. Used to determine outliers based on the Mean Absolute Deviation - Median criteria.
