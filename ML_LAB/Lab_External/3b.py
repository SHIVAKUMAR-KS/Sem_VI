import statistics
# Sample data for the demonstration
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
grouped_data = [12, 16, 14, 18, 22, 24, 21, 28]
# Harmonic Mean
harmonic_mean = statistics.harmonic_mean(data)
# Mean
mean_value = statistics.mean(data)
# Median
median_value = statistics.median(data)
# Median of Grouped Data
median_grouped = statistics.median_grouped(grouped_data)
# High Median
high_median = statistics.median_high(data)
# Low Median
low_median = statistics.median_low(data)
# Mode (mode for sample data)
mode_value = statistics.mode([1, 2, 2, 3, 4, 5, 6, 7, 8, 9])
# Population Standard Deviation
population_stdev = statistics.pstdev(data)
# Sample Standard Deviation
sample_stdev = statistics.stdev(data)
# Population Variance
population_variance = statistics.pvariance(data)
# Sample Variance
sample_variance = statistics.variance(data)
# Output all the results
print("Harmonic Mean:", harmonic_mean)
print("Mean:", mean_value)
print("Median:", median_value)
print("Median Grouped:", median_grouped)
print("High Median:", high_median)
print("Low Median:", low_median)
print("Mode:", mode_value)
print("Population Standard Deviation:", population_stdev)
print("Sample Standard Deviation:", sample_stdev)
print("Population Variance:", population_variance)
print("Sample Variance:", sample_variance)