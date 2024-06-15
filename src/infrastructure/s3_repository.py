from interfaces.repository import Repository
from models.feedback import Feedback
import json
import boto3


class S3Repository(Repository):
    def __init__(self):
        self.s3 = boto3.client('s3')

    def AddFeedback(self, feedback: Feedback) -> None:
        """Add a feedback item to the S3 bucket."""
        serialized_feedback = json.dumps(feedback.__dict__, default=str)
        self.s3.put_object(Body=serialized_feedback, Bucket='fao-ppa-classifier-feedback',
                           Key=feedback.timestamp.strftime('%Y_%m_%d_%H_%M_%S_%f.json'))
