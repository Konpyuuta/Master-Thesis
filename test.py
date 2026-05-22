'''from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

from imputegap.recovery.contamination import GenGap

# initialize the time series object
ts = TimeSeries()

# load and normalize the dataset
ts.load_series(utils.search_path("bafu"))

# contaminate the time series
ts.data = ts.data.reshape(-1, 12)
print(ts.data.shape)
ts_m = GenGap.mcar(ts.data, block_size=2, rate_dataset=0.02, rate_series=0.02)

# impute the contaminated series
imputer = Imputation.LLMs.NuwaTS(ts_m)
imputer.impute()

# compute and print the imputation metrics
imputer.score(ts.data, imputer.recov_data)
ts.print_results(imputer.metrics)

# plot the recovered time series
ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputer.recov_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")'''

from imputegap.recovery.benchmark import Benchmark
from imputegap.tools import utils

my_algorithms = ["llmts"]

my_opt = "default_params"

my_datasets = ["forecast-economy"]

my_patterns = ["mcar"]

range = [0.125]
# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range)