from google.cloud import aiplatform

def main():
    # Define your GCP project, region, and Cloud Storage bucket
    project_id = "bigdatainpractice1"
    region = "us-central1"
    bucket = "gs://27arp2023t2042"
    display_name = "custom-training-job"
    container_image = "gcr.io/cloud-aiplatform/training/tf-cpu.2-1:latest"

    # Set the local path of the training script
    local_code_path = "train.py"

    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=region, staging_bucket=bucket)

    # Create and run a custom training job
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path=local_code_path,
        container_uri=container_image,
        requirements=["tensorflow==2.1.0"],
    )

    # Train the model
    model = job.run(
        replica_count=1,
        machine_type="n1-standard-4",
    )

if __name__ == "__main__":
    main()
