# Synthetic cube data

80 fictional paired observations, 40 batches, two sample pairs per batch. Each pair
represents distinct specimens tested at 7 and 28 days, not the same cube retested.
Both columns are cube compressive strength in MPa. No cylinder/cube conversion occurs.

Python random.Random(751): for each batch draw early~Uniform(14,38) and batch
effect~Normal(0,2.5). For each of two pairs draw x=early+Normal(0,0.7), round to
two decimals; y=7+1.23*x+batch_effect+Normal(0,1.2), round to two decimals.
This invented relationship is deliberately learnable. It is not a concrete law.
The included generator reproduces the CSV without accessing any external data.

GroupShuffleSplit(seed=42, test_size=.25) keeps both pairs from a batch together.
This checks batch separation but does not establish generalisation to new mixes,
plants, curing regimes, weather, sites or future dates. The public test split is
a demonstration, not a secret benchmark. No tuning was performed on its results.
