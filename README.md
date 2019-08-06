# stats_test_from_scratch
One issue with Python is that there is no unified source for statistical tests like there is for ML with scikit-learn or Deep Learning with Keras. 

You could have statistical tests in Scipy, Statsmodels, Numpy and Pandas; but it's not clear which libraries carry which tests. Additionally, there are some tests that you can find in R that aren't currently supported by these libraries.

My goal is two fold:
1) Implement the statistical tests from scratch, with an emphasis on making the code as presentable and easy-to-understand as possible. This is so anyone can understand exactly how what the test is measuring and how it is measuring it, even if it means we are sacrificing computational speed in the process.
2) Identify where in Python you can find these test(s), if at all. That way, for those who want the fastest implementation, they'll understand where to find the test.

# Statistical Tests currently supported and where to find them:
## Sample Tests
1) One and two sample Z Tests: Statsmodels through [ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.ztest.html).
2) One and two sample T Tests: Scipy through [ttest_1samp](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.ttest_1samp.html) and [ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
3) One and two sample Proportion Z Tests: Statsmodels through  [proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)
4) Two Sample F Test: Not found in either scipy or statsmodels.

## Rank Tests
1) Wilcoxon Rank-Sum Test: Scipy through [wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
2) Mann-Whitney-U Test: Scipy through [mannwhitneyu](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html)
3) Friedman Test: Scipy through [friedmanchisquare](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html)
4) Page's Trend Test: Not found in either scipy or statsmodels.
5) Kruskal-Wallis Test: Scipy through [kruskal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html)

## Categorical Tests
1) Chi Square Test: Scipy through [chi2_contingency](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html)
2) G Test: Scipy through [chi2_contingency(lambda_="log-likelihood")](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.chi2_contingency.html)
3) Chi Goodness of Fit Test: Scipy through [chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html)
4) G Goodness of Fit Test: Scipy through [power_divergence(lambda_="log-likelihood")](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.power_divergence.html)
5) Fisher Test: Scipy through [fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
6) McNemar Test: Statsmodels through [mcnemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html)
7) Cochran–Mantel–Haenszel Test: Statsmodels through [StratifiedTable.test_null_odds]https://www.statsmodels.org/dev/generated/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds.html#statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds)

## Multi-Group Tests
1) Levene Test: Scipy through [levene(center='mean')](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html)
2) Brown-Forsythe Test: Scipy through [levene(center='median')](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.levene.html)
3) One Way F-Test: Scipy through [f_oneway](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
4) Bartlett Test: Scipy through [bartlett](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html)
5) Tukey Range Test: Statsmodels through [pairwise_tukeyhsd](https://www.statsmodels.org/stable/generated/statsmodels.stats.multicomp.pairwise_tukeyhsd.html)
6) Cochran's Q Test: Statsmodels through [cochrans_q](https://www.statsmodels.org/devel/generated/statsmodels.stats.contingency_tables.cochrans_q.html)
7) Jonckheere Trend Test: Not found in either scipy or statsmodels
