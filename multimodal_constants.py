
IMAGE = [
        "image_h",
        "image_w",
        "num_boxes",
        "boxes",
        "features",
        "class_labels"
    ]

QUERY = [
        "query",
        "query_id"
    ]

PRODUCT = [
        "product_id"
    ]

LABEL_KEY = "relevancy"

# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
