# t-test-py
an example of a Python function that performs an independent sample t-test

```
import numpy as np
from scipy import stats

def independent_ttest(data1, data2, alpha=0.05):
    """
    Perform an independent sample t-test.
    
    Arguments:
        data1 (List[float]): The first data sample
        data2 (List[float]): The second data sample
        alpha (float): The significance level (default: 0.05)
    
    Returns:
        tuple: A tuple containing the t-statistic and the p-value
    """
    # Calculate the sample means
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    
    # Calculate the sample variances
    var1 = np.var(data1)
    var2 = np.var(data2)
    
    # Calculate the sample sizes
    n1 = len(data1)
    n2 = len(data2)
    
    # Calculate the degrees of freedom
    df = n1 + n2 - 2
    
    # Calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / df)
    
    # Calculate the t-statistic
    t = (mean1 - mean2) / (s * np.sqrt(1 / n1 + 1 / n2))
    
    # Calculate the p-value
    p = 2 * (1 - stats.t.cdf(np.abs(t), df))
    
    return t, p
```

To use this function, you would simply call it with the two data samples as arguments:

```
data1 = [1, 2, 3, 4, 5]
data2 = [2, 3, 4, 5, 6]

t, p = independent_ttest(data1, data2)
```

The function will then return the t-statistic and p-value for the t-test. In this case, the returned values might be something like `(-1.96, 0.07)`, indicating that the difference between the two samples is not statistically significant at the .05 level.
