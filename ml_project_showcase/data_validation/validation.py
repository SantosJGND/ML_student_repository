import pandas as pd
import pandera as pa
from config import get_logger
from data_validation.schemas import BreastCancerRecord, BreastCancerSchema

logger = get_logger(__name__)


REQUIRED_COLS = list(BreastCancerSchema.columns.keys())


def validate_with_pydantic(df):
    """
    Validate each row of a DataFrame against the Pydantic BreastCancerRecord model.
    Args:
        df: DataFrame to validate
    Returns:
        Tuple: (valid DataFrame, invalid DataFrame with error messages)
    """
    valid_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        try:
            record = BreastCancerRecord(**row.to_dict())
            valid_rows.append(row)
        except Exception as e:
            row_with_error = row.to_dict()
            row_with_error["_validation_error"] = str(e)
            invalid_rows.append(row_with_error)

    valid_df = pd.DataFrame(valid_rows) if valid_rows else pd.DataFrame(columns=df.columns)
    invalid_df = pd.DataFrame(invalid_rows) if invalid_rows else pd.DataFrame()

    logger.info(f"Pydantic validation: {len(valid_df)} valid, {len(invalid_df)} invalid (out of {len(df)})")
    return valid_df, invalid_df


def validate_with_pandera(df):
    """
    Validate a DataFrame against the Pandera BreastCancerSchema.
    Args:
        df: DataFrame to validate
    Returns:
        Tuple: (bool success, str or None error_message)
    """
    try:
        BreastCancerSchema.validate(df, lazy=True)
        logger.info("Pandera validation: PASSED")
        return True, None
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Pandera validation: FAILED ({e})")
        return False, str(e)
    except Exception as e:
        logger.error(f"Pandera validation error: {e}")
        return False, str(e)
