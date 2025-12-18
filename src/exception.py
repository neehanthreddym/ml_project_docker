import sys
# import logging
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info() # Get exception info
    error_message = f"Error occured in script: {exc_tb.tb_frame.f_code.co_filename}" \
                    f" \nat line number: {exc_tb.tb_lineno}" \
                    f" \nError message: {str(error)}"
    logging.error(error_message)
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

# if __name__ == "__main__":
#     try:
#         1 / 0
#     except Exception as e:
#         logging.info("Division by zero error caught.")
#         raise CustomException(e, sys)