# Azure Functions Trace 2019 dataset

Dataset on Kaggle[1], GitHub[2].

Download and extract the dataset to `data` directory with:

```
$ wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
$ unxz azurefunctions-dataset2019.tar.xz
$ mkdir data
$ tar --get --file azurefunctions-dataset2019.tar --directory data
```

There are three types of CSV files extracted from the tarball. I am only
interested in the one named `invocations_per_function_md.anon.dXX.csv`.

[1]: https://www.kaggle.com/datasets/theodoram/azure-2019-public-dataset
[2]: https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
