import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# TODO: Import your schemas here
# from data_validation.schemas import IrisRecord, IrisSchema


# TODO: Write tests here
# def test_iris_record_negative_sepal_length():
#     try:
#         IrisRecord(sepal_length=-1, sepal_width=3.0, petal_length=1.0, petal_width=0.5, class_=0)
#         assert False, "Should raise ValueError"
#     except ValueError:
#         pass


# def test_iris_record_valid():
#     record = IrisRecord(sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2, class_=0)
#     assert record.sepal_length == 5.1


# TODO: Add more tests for:
# - Invalid class values (not 0, 1, or 2)
# - Invalid petal measurements
# - Pandera schema validation
