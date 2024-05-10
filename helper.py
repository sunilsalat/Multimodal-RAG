

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)


def upload_img_to_s3(image_path):
    with open(image_path, "rb") as f:
      s3.upload_fileobj(f, S3_BUCKET_NAME, image_path)
    return image_path



def read_img_from_s3(image_key):
    image_url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': S3_BUCKET_NAME, 'Key': image_key},
        ExpiresIn=3600  # URL expiration time in seconds (optional)
    )
    return image_url


def upload_batch_images_to_s3(output_path):
    for i in os.listdir(output_path):
        image_path = os.path.join(output_path, i)
        upload_img_to_s3(image_path)


def read_batch_images_to_s3(output_path):
    img_urls = []
    for i in os.listdir(output_path):
        image_path = os.path.join(output_path, i)
        read_img_from_s3(image_path)
    return img_urls

__all__ = ['upload_img_to_s3', 'read_img_from_s3', 'upload_batch_images_to_s3', 'read_batch_images_to_s3' ]
