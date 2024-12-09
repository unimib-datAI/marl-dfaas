# Azure Functions Trace 2019 dataset

These traces are used to train the agents with real data instead of synthetic
data. I found this dataset on
[Kaggle](https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset)
and there is a dedicated repository on
[GitHub](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md).
The dataset is not provided by default, you need to download it, extract and
process it before starting a new training or evaluation experiment:

```
$ wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
$ unxz azurefunctions-dataset2019.tar.xz
$ mkdir data
$ tar --get --file azurefunctions-dataset2019.tar --directory data
```

Then run `filter_requests.py` to process the requests:

    $ python dataset/filter_requests.py

The process is necessary to carry out the experiments with these traces, since
the original traces cannot be used directly (they are oversized both in time and
in invocations).

There are three types of CSV files extracted from the tarball. I am only
interested in the one named `invocations_per_function_md.anon.dXX.csv`.
