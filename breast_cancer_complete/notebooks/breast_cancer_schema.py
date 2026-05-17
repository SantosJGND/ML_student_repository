import pandera as pa

BreastCancerSchema = from pandera import DataFrameSchema, Column, Check, Index, MultiIndex

schema = DataFrameSchema(
    columns={
        "Diagnosis": Column(
            dtype="int64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=1.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Radius": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=6.981, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=388.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Radius": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=9.71, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=39.28, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Radius": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=43.79, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=188.5, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Texture": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=143.5, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=28176.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Texture": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.05263, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.1634, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Texture": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.01938, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.3454, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Perimeter": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=6.8288, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Perimeter": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.2012, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Perimeter": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.106, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.304, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Area": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.04996, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=1.24512, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Area": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.1115, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=2.873, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Area": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.3602, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=4.885, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Smoothness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.757, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=160.8, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Smoothness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=6.802, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=542.2, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Smoothness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.001713, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.03113, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Compactness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.002252, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=1.56896, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Compactness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.396, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Compactness": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.05279, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Concavity": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.007882, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.80224, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Concavity": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0008948, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.02984, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Concavity": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=7.93, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=36.04, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_ConcavePoints": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=12.02, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=570.24, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_ConcavePoints": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=50.41, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=251.2, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_ConcavePoints": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=185.2, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=4254.0, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_Symmetry": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.07117, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=3.0048, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_Symmetry": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.02729, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=1.058, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_Symmetry": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=1.252, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Mean_FractalDimension": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.0, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=4.656, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "SE_FractalDimension": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.1565, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.6638, raise_warning=False, ignore_na=True
                ),
            ],
        ),
        "Worst_FractalDimension": Column(
            dtype="float64",
            checks=[
                Check.greater_than_or_equal_to(
                    min_value=0.05504, raise_warning=False, ignore_na=True
                ),
                Check.less_than_or_equal_to(
                    max_value=0.2075, raise_warning=False, ignore_na=True
                ),
            ],
        ),
    },
    index=Index(
        dtype="int64",
        checks=[
            Check.greater_than_or_equal_to(
                min_value=0.0, raise_warning=False, ignore_na=True
            ),
            Check.less_than_or_equal_to(
                max_value=568.0, raise_warning=False, ignore_na=True
            ),
        ],
        nullable=False,
        coerce=False,
        name=None,
        description=None,
        title=None,
    ),
    coerce=True,
)
