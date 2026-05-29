import json

def ensure_newlines(source_list):
    """Ensure each non-empty source line ends with \\n."""
    return [s if s.endswith("\n") or s == "" else s + "\n" for s in source_list]

with open("notebooks/02_data_validation.ipynb") as f:
    nb = json.load(f)

# Normalize ALL code cells first (fix missing \\n issue)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["source"] = ensure_newlines(cell["source"])

FEATURE_COLS = [
    "Mean_Radius", "SE_Radius", "Worst_Radius",
    "Mean_Texture", "SE_Texture", "Worst_Texture",
    "Mean_Perimeter", "SE_Perimeter", "Worst_Perimeter",
    "Mean_Area", "SE_Area", "Worst_Area",
    "Mean_Smoothness", "SE_Smoothness", "Worst_Smoothness",
    "Mean_Compactness", "SE_Compactness", "Worst_Compactness",
    "Mean_Concavity", "SE_Concavity", "Worst_Concavity",
    "Mean_ConcavePoints", "SE_ConcavePoints", "Worst_ConcavePoints",
    "Mean_Symmetry", "SE_Symmetry", "Worst_Symmetry",
    "Mean_FractalDimension", "SE_FractalDimension", "Worst_FractalDimension",
]

# ── Cell [2]: Load clean data ──
nb["cells"][2]["source"] = [
    'PROCESSED_DIR = Path("../data/processed")\n',
    "\n",
    'df = pd.read_csv(PROCESSED_DIR / "clean_data.csv")\n',
    'print(f"Data shape: {df.shape}")\n',
    'print(f"Columns: {list(df.columns)}")\n',
]

# ── Cell [5]: BreastCancerRecord Pydantic model ──
model_source = [
    "class BreastCancerRecord(BaseModel):\n",
    '    """Pydantic model for a single breast cancer record."""\n',
    "\n",
    "    Diagnosis: int\n",
]
for col in FEATURE_COLS:
    model_source.append(f"    {col}: float\n")
model_source.append("\n")
model_source.append('    @field_validator("Diagnosis")\n')
model_source.append("    @classmethod\n")
model_source.append("    def diagnosis_binary(cls, v):\n")
model_source.append('        """Diagnosis must be 0 (benign) or 1 (malignant)."""\n')
model_source.append("        if v not in [0, 1]:\n")
model_source.append('            raise ValueError(f"Diagnosis must be 0 or 1, got {v}")\n')
model_source.append("        return v\n")
model_source.append("\n")
model_source.append('    @field_validator("Mean_Radius")\n')
model_source.append("    @classmethod\n")
model_source.append("    def radius_positive(cls, v):\n")
model_source.append('        """Mean_Radius must be within realistic range."""\n')
model_source.append("        if not (0 < v <= 30):\n")
model_source.append('            raise ValueError(f"Mean_Radius out of range: {v}")\n')
model_source.append("        return v\n")
model_source.append("\n")
model_source.append("    # Add similar validators for other feature columns:\n")
model_source.append("    # - All Mean_* and Worst_* : must be positive\n")
model_source.append("    # - All SE_* : must be non-negative\n")
model_source.append("    # - Mean_FractalDimension, SE_FractalDimension, Worst_FractalDimension: 0 to 1\n")

nb["cells"][5]["source"] = model_source

# ── Cell [6]: validate_with_pydantic ──
nb["cells"][6]["source"] = [
    "def validate_with_pydantic(df):\n",
    '    """Validate each row of the DataFrame using the Pydantic model.\n',
    "\n",
    "    Returns a list of booleans where True means the row is valid.\n",
    '    """\n',
    "    valid = []\n",
    "    for _, row in df.iterrows():\n",
    "        try:\n",
    "            BreastCancerRecord(**row.to_dict())\n",
    "            valid.append(True)\n",
    "        except Exception:\n",
    "            valid.append(False)\n",
    "\n",
    "    valid_count = sum(valid)\n",
    "    invalid_count = len(valid) - valid_count\n",
    '    print(f"Pydantic: {valid_count} valid, {invalid_count} invalid out of {len(valid)} rows")\n',
    "    return valid\n",
    "\n",
    "\n",
    "valid_flags = validate_with_pydantic(df)\n",
    'print(f"Pass rate: {sum(valid_flags) / len(valid_flags):.1%}")\n',
]

# ── Cell [7]: Replace with pure comments (no active code) ──
nb["cells"][7]["source"] = [
    "# === Commented Template: Pydantic Row-Level Validation ===\n",
    "# Copy the BreastCancerRecord class above and adapt it to your own dataset.\n",
    "# Uncomment the code below to validate any DataFrame row-by-row:\n",
    "#\n",
    "# def validate_with_pydantic(df):\n",
    "#     valid = []\n",
    "#     for _, row in df.iterrows():\n",
    "#         try:\n",
    "#             BreastCancerRecord(**row.to_dict())\n",
    "#             valid.append(True)\n",
    "#         except Exception:\n",
    "#             valid.append(False)\n",
    "#     return valid\n",
    "#\n",
    "# valid_flags = validate_with_pydantic(df)\n",
    "# print(f'Pass rate: {sum(valid_flags) / len(valid_flags):.1%}')\n",
]

# ── Cell [9]: Pandera DataFrameSchema ──
pandera_schema_lines = [
    "import warnings\n",
    "warnings.filterwarnings('ignore', category=DeprecationWarning)\n",
    "\n",
    "feature_cols = [\n",
]
for col in FEATURE_COLS:
    pandera_schema_lines.append(f'    "{col}",\n')
pandera_schema_lines.append("]\n")
pandera_schema_lines.append("\n")
pandera_schema_lines.append("BreastCancerSchema = pa.DataFrameSchema({\n")
pandera_schema_lines.append('    "Diagnosis": pa.Column(pa.Int, checks=pa.Check.isin([0, 1])),\n')
pandera_schema_lines.append("    **{\n")
pandera_schema_lines.append("        col: pa.Column(pa.Float, checks=pa.Check.ge(0))\n")
pandera_schema_lines.append("        for col in feature_cols\n")
pandera_schema_lines.append("    },\n")
pandera_schema_lines.append("})\n")
pandera_schema_lines.append("\n")
pandera_schema_lines.append('print("BreastCancerSchema defined with all 31 columns")\n')

nb["cells"][9]["source"] = pandera_schema_lines

# ── Cell [10]: validate_with_pandera ──
nb["cells"][10]["source"] = [
    "def validate_with_pandera(df):\n",
    '    """Validate the entire DataFrame using the Pandera schema."""\n',
    "    try:\n",
    "        BreastCancerSchema.validate(df, lazy=True)\n",
    '        print("Pandera: all rows passed validation")\n',
    "    except pa.errors.SchemaErrors as e:\n",
    '        print(f"Pandera: {len(e.failure_cases)} failure(s) found")\n',
    '        print(e.failure_cases[["index", "column", "check", "failure_case"]].head())\n',
    "\n",
    "\n",
    "validate_with_pandera(df)\n",
]

# ── Cell [11]: Corrupted data test ──
nb["cells"][11]["source"] = [
    "# Load corrupted data with missing values in Diagnosis\n",
    'df_corrupted = pd.read_csv(PROCESSED_DIR / "corrupted" / "corrupted_missing_light.csv")\n',
    f"print(f\"Corrupted data shape: {{df_corrupted.shape}}\")\n",
    f"print(f\"Missing values per column:\\n{{df_corrupted.isnull().sum()}}\\n\")\n",
    "\n",
    "# Pydantic validation on corrupted data\n",
    'print("--- Pydantic Validation ---")\n',
    "valid_corrupted = validate_with_pydantic(df_corrupted)\n",
    "cv = sum(valid_corrupted)\n",
    "ct = len(valid_corrupted)\n",
    'print(f"Corrupted pass rate: {cv}/{ct} ({cv/ct:.1%})")\n',
    "\n",
    "# Pandera validation on corrupted data\n",
    'print("\\n--- Pandera Validation ---")\n',
    "validate_with_pandera(df_corrupted)\n",
]

# ── Cell [12]: Comment out old function def so it doesn't override cell [10] ──
src12 = nb["cells"][12]["source"]
src12 = [s.replace("def validate_with_pandera", "# def validate_with_pandera")
         for s in src12]
src12 = [s.replace("    pass", "    # pass") for s in src12]
nb["cells"][12]["source"] = src12

# ── Save ──
with open("notebooks/02_data_validation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully")
