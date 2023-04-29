from google.cloud import aiplatform

# Set your project ID, region, and the GCS bucket
project_id = "bigdatainpractice1"
region = "us-central1"
gcs_bucket = "29apr2023"

# Set the GCS path for the job output
#job_folder = f"gs://{gcs_bucket}/output"
job_folder = f"gs://{gcs_bucket}/output/model"


# Initialize the Vertex AI SDK with the staging bucket
aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{gcs_bucket}/staging")

# Submit the training job to Vertex AI
job = aiplatform.CustomJob.from_local_script(
    display_name="mnist-training",
    script_path="run_train_module.py",  # Use the run_train_module.py script
    requirements=None,  # No additional requirements since we're using a custom container
    container_uri=f"gcr.io/{project_id}/mnist_training_container",  # Use the custom container
    args=[
        "--job-dir",
        job_folder,
    ],
)

job.run(sync=True)

# Create and deploy a Model object with the trained model
model = aiplatform.Model.upload(
    display_name="mnist-model",
    artifact_uri=job_folder,  # Model artifacts are in the job_folder
    serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-6:latest",  # Use the TensorFlow 2.6 serving image
)

# Deploy the model to an endpoint
endpoint = model.deploy(machine_type="n1-standard-4")

