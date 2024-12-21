import boto3
import json

def test_textract_access():
    try:
        # Create Textract client
        textract = boto3.client('textract')
        print("Successfully created Textract client!")
        
        # Get service info (lightweight call to verify access)
        response = textract.list_document_text_detection_jobs(
            MaxResults=1
        )
        
        print("\nTextract Access Verified!")
        print(f"Response: {json.dumps(response, indent=2)}")
        return True
        
    except Exception as e:
        print(f"Error accessing Textract: {str(e)}")
        return False

if __name__ == "__main__":
    test_textract_access()