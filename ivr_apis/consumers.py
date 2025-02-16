from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
import re, json, os, zipfile, time, logging, csv, secrets, whisper, torch, traceback, warnings, asyncio, sqlite3, base64, tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import dateparser

# Ignore all warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
# Set the CUDA_LAUNCH_BLOCKING environment variable
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Loading Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
whisper_model = whisper.load_model(str(settings.BASE_DIR)+"/model/small.en.pt",  device=device)
# # Define the model and tokenizer
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

class IVRConsumer(AsyncWebsocketConsumer):
    
    async def connect(self):
        await self.accept()
        self.patient_account_number = None  # Stores extracted patient account number
        self.initialinput = True
        self.awaiting_confirmation = False
        self.awaiting_payment = False
        self.awaiting_card_number = False
        self.awaiting_verification = False
        self.awaiting_first_name = False
        self.session_data = {}  # Store ongoing session data (account number, balance, etc.)
        self.awaiting_expiry_date = False
        self.awaiting_cvv = False
        self.expiry_date = None
        self.card_number = None
        self.expiry_date = None
        self.cvv = None
        self.awaiting_card_confirmation = False
        self.awaiting_after_confirmation = False
        self.awaiting_account_confirmation = False
        await self.send(json.dumps({"status": "Connected", "message": "IVR session started. Please provide your Last Name(with spelling) and DOB."}))

    async def is_valid_account_number(self, account_number):
        """
        Validates the extracted account number using regex.
        """
        account_number_pattern = (
            r"^(?:4[0-9]{12}(?:[0-9]{3})?"            # Visa
            r"|5[1-5][0-9]{14}"                      # MasterCard (Old Range)
            r"|2(?:2[2-9][0-9]{2}|[3-6][0-9]{3}|7[0-1][0-9]{2}|720)[0-9]{12}"  # MasterCard (New Range)
            r"|3[47][0-9]{13}"                       # American Express
            r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"      # Diners Club
            r"|6(?:011|5[0-9]{2}|22(?:12[6-9]|1[3-9][0-9]|[2-8][0-9]{2}|9[0-1][0-9]|92[0-5]))[0-9]{12}"  # Discover
            r"|(?:2131|1800|35\d{3})\d{11})$"       # JCB
        )
        return bool(re.match(account_number_pattern, account_number))
    
    async def extract_data(self, user_input, bit):
        print("BIT VALUE IS: ", bit)
        print("USER INPUT IS: ", user_input)
        pt_acc_prompt = f"""
            Extract and present the patient account number from the user's input in the specified format:
            "xxxxxxxxxxxxxxxx"
            Please extract the patient account number and present it in the specified format,
            making sure to replace the "x" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        """
        
        cc_prompt = f"""
            Extract and present the credit card number from the user's input in the specified format:
            "xxxxxxxxxxxxxxxx"
            Please extract the credit card number and present it in the specified format,
            making sure to replace the "x" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        """
 
        date_prompt = f"""
            You are a strict date formatter. You will receive a credit card expiry date in various formats and must convert it to MM/YY format.
            Rules:
            Month must be two digits (e.g., January → 01, Feb → 02, ..., December → 12).
            Year must be the last two digits (e.g., 2025 → 25, 2029 → 29).
            DO NOT invent dates—only reformat what is given.
            If input is already in MM/YY format, do not change it.
            Output must always be in MM/YY format and nothing else.
            Examples:
            "April 2026" → "04/26"
            "November 29" → "11/29"
            "Feb 2025" → "02/25"
            "10 26" → "10/26"

            Now, reformat this date:
            "{user_input}"
                        """

        cvc_prompt = f"""
            Extract and present the Card Verification Code (CVC) / Card Verification Value (CVV) from the user's input in the specified format:
            "XXX"
            Please extract the CVC/CVV and present it in the specified format,
            making sure to replace the "XXX" with the actual numbers you extracted from the user's input.
            Convert all words to numerical form accurately:
            Text: "{user_input}"
 
            Output only the extracted and formatted number. Do not include any additional questions or answers.
        # """
        custom_amount_prompt = f"""
        Extract and numerical figure from provided text in alphabetical form and present it in the specified format:
        "XX.XX"
        making sure to replace the "XX.XX" with the actual numbers you extracted from the user's input.
        Convert all words to numerical form accurately:
        Text: "{user_input}"

        """

        DOB_prompt = f"""
            Extract the expiry date from the user's input and present it in the "DD/MM/YY" format:
            - "DD" is the two-digit numerical representation of the date (e.g., "Sixth" becomes "06", "twentyfirst" becomes "21")
            - "MM" is the two-digit numerical representation of the month (e.g., "January" becomes "01", "March" becomes "03").
            - "YYYY" is the year of birth in four digits (e.g., "Twenty Twenty-Five" becomes "2025").
            
            Important Guidelines:
            1. If the input contains only numbers (e.g., "7 25"), interpret them as "DD MM YY".
            2. Ignore any text that is not part of the date.

            Text: "{user_input}"

            Output only the extracted and formatted date in "MM/YY" format. Do not include explanations, additional text, or questions.
        """

        name_extract = f"""
            You are a precise extractor. Your ONLY task is to extract the last name from user input, ensuring correct spelling.

            Rules:
            The last name is always spelled out after "last name is" or a similar phrase.
            If spelled out, use the spelled-out version (not the spoken one).
            Return only the last name—no extra words, explanations, or formatting.
            Case-sensitive: Maintain capitalization exactly as spelled.
            DO NOT guess or modify the last name—just return what is spelled out.
            Example Inputs & Correct Outputs:
            "Okay, last name is Colin, C-O-L-L-I-N, and date of birth is 19th December 1992." → "Collin"
            "Last name's David- sorry, Davidson. D-A-V-I-D-S-O-N." → "Davidson"
            "Sure, last name is Lee." → "Lee"
            "His surname is Smith, S-M-I-T-H." → "Smith"

            Now, extract the last name from this:

            Text: "{user_input}"

            Do not include explanations, additional text, or questions.
        """

        prompt = ""
        if bit==0:
            prompt = pt_acc_prompt
        elif bit==1:
            prompt = cc_prompt
        elif bit==2:
            prompt = date_prompt
        elif bit==3:
            prompt = cvc_prompt
        elif bit==4:
            prompt = custom_amount_prompt
        elif bit==5:
            prompt = DOB_prompt
        elif bit==6:
            prompt = name_extract
 
        # Construct the full prompt
        full_prompt = f"You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nUser: {prompt}\nAssistant:"
 
        # Tokenize the input
        model_inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        await asyncio.sleep(0)  # Yield control
        print(device)
 
        # Generate the response
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,  
            temperature=0.1,     
            top_p=0.95,          
            do_sample=False      
        )
 
        # Decode and clean the generated text
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        response  = response.split('Assistant:')[1].strip()
        print(f"response: {response}")
        if bit == 1 and not await self.is_valid_account_number(response):
            return {"status": "Failed", "message": "Account number is not valid"}
        return {"status": "Success", "message": response}
    
    async def receive(self, text_data=None, bytes_data=None):
        """Handles incoming WebSocket messages (audio chunks)"""
        data = json.loads(text_data)
        print(data)
        if "audio_chunk" in data and "type" in data:
            transcription_result = self.transcribe_audio_from_base64(data["audio_chunk"])
            if transcription_result is None:
                await self.send(json.dumps({"error": "Failed to transcribe audio."}))
                return

            transcribed_text = transcription_result.get("text", "")
            print("Transcribed text:", transcribed_text)
            if self.awaiting_verification:
                await self.process_account_number(transcribed_text)
            elif self.awaiting_first_name:
                await self.handle_first_name(transcribed_text)
            elif self.awaiting_confirmation:
                await self.handle_confirmation(transcribed_text)
            elif self.awaiting_payment:
                await self.handle_payment_choice(transcribed_text)
            elif self.awaiting_card_number:
                await self.handle_card_details(transcribed_text)
            elif self.awaiting_expiry_date:
                await self.handle_expiry_date(transcribed_text)
            elif self.awaiting_cvv:
                await self.handle_cvv(transcribed_text)
            elif self.awaiting_after_confirmation:
                await self.handle_card_confirmation(transcribed_text)
            elif self.initialinput:
                await self.handle_initial_input(transcribed_text)
            elif self.awaiting_account_confirmation:
                await self.handle_account_confirmation(transcribed_text)
    
    async def handle_account_confirmation(self, transcribed_text):
        """Processes Yes/No response for account verification"""
        confirmation = self.process_confirmation(transcribed_text)
        await self.send(json.dumps({"confirmation_response": confirmation}))

        if confirmation == "yes":
            if self.verify_account_number(self.patient_account_number):
                await self.fetch_and_prompt_due_amount()
            else:
                await self.send(json.dumps({"message": "Account number not found. Please re-enter."}))
                self.patient_account_number = None
        else:
            await self.send(json.dumps({"message": "Account number incorrect. Please re-enter."}))
            self.patient_account_number = None
        
        self.awaiting_confirmation = False

    async def handle_initial_input(self, transcribed_text):
        """Extract last name and DOB, and verify in the database."""
        data_type = 6
        last_name_data = await self.extract_data(transcribed_text, data_type)
        last_name = last_name_data.get("message") if isinstance(last_name_data, dict) else last_name_data
        dob = await self.extract_and_format_date(transcribed_text)
        print(last_name)
        print(dob)
        if last_name and dob:
            matching_records = await self.verify_patient_lastname_dob(last_name, dob)
            if len(matching_records) == 1:
                self.patient_account_number = matching_records[0]["account_number"]
                await self.send(json.dumps({"message": "Patient verified. Fetching payment details."}))
                await self.fetch_and_prompt_due_amount()
                self.initialinput = False
            elif len(matching_records) > 1:
                await self.send(json.dumps({"message": "Multiple matches found. Please provide your first name."}))
                self.awaiting_first_name = True
            else:
                await self.send(json.dumps({"message": "No match found. Please provide your patient account number."}))
                self.awaiting_verification = True
        else:
            await self.send(json.dumps({"error": "Could not detect last name or date of birth. Please try again."}))
            self.initialinput = True

    async def extract_and_format_date(self, text):
        """Extracts a date from a sentence and converts it to DD/MM/YYYY format."""
        # Regular expression to find date-like patterns in text
        date_pattern = r"(\d{1,2}(st|nd|rd|th)?\s+(of\s+)?[A-Za-z]+\s+\d{4}|\b[A-Za-z]+\s+\d{1,2}(st|nd|rd|th)?,?\s+\d{4})"
        match = re.search(date_pattern, text)

        if match:
            date_text = match.group(0)  # Extract matched date portion
            date_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_text)  # Remove ordinal suffixes
            parsed_date = dateparser.parse(date_text)  # Convert to datetime object
            
            if parsed_date:
                return parsed_date.strftime("%d/%m/%Y")  # Convert to DD/MM/YYYY format
        
        return None   

    def get_db_connection(self):
        conn = sqlite3.connect("/home/adam/adam/IVR/Patient_Payment_IVR/ivr_apis/patients.db")
        conn.row_factory = sqlite3.Row  # This makes rows behave like dictionaries
        return conn
    
    async def verify_patient_lastname_dob(self, last_name: str, dob: str):
        """
        Verify if a patient exists with the given last name and date of birth (DOB).

        :param last_name: The last name of the patient.
        :param dob: The date of birth of the patient (DD/MM/YYYY format).
        :return: A list of matching patient records or an empty list if no match is found.
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()
        try:
            query = "SELECT * FROM patients WHERE last_name = ? AND dob = ?"
            cursor.execute(query, (last_name, dob))
            results = cursor.fetchall()

            return results  # Returns a list of tuples

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []

        finally:
            conn.close()


    async def handle_first_name(self, transcribed_text):
        """Extract and verify first name if multiple matches exist."""
        first_name = await self.extract_firstname(transcribed_text)
        
        if first_name:
            matching_records = self.verify_patient_fullname(first_name, self.last_name, self.dob)
            if len(matching_records) == 1:
                self.patient_account_number = matching_records[0]["account_number"]
                await self.send(json.dumps({"message": "Patient verified. Fetching payment details."}))
                await self.fetch_payment_details()
            else:
                await self.send(json.dumps({"message": "Still multiple or no matches. Please provide your patient account number."}))
                self.awaiting_verification = True
        else:
            await self.send(json.dumps({"error": "Could not detect first name. Please try again."}))


    async def fetch_and_prompt_due_amount(self):
        """Fetches due amount from DB and prompts for payment"""
        due_amount = self.fetch_due_amount(self.patient_account_number)
        self.due_amount = due_amount
        await self.send(json.dumps({"message": f"Your total due balance is ${due_amount}. Would you like to pay this amount or a different amount?"}))
        self.awaiting_payment = True
        self.initialinput = False

    async def handle_payment_choice(self, transcribed_text):
        """Handles whether patient wants to pay full or a different amount"""
        if self.process_confirmation(transcribed_text) == "yes":
            self.payment_amount = self.due_amount
        else:
            try:
                bit = 4
                other_amount = await self.extract_data(transcribed_text, bit)
                print("EXTRACTED OTHER AMOUNT FROM AUDIO: ", other_amount)
                other_amount = other_amount.get("message")
                self.payment_amount = float(other_amount)
                await self.send(json.dumps({"message": f"Proceeding with payment of ${self.payment_amount:.2f}."}))
            except ValueError:
                await self.send(json.dumps({"error": "Invalid amount received. Please try again."}))
                return

        await self.send(json.dumps({"message": "We accept all major credit and debit cards. Please share your 16-digit card number."}))
        self.awaiting_card_number = True
        self.awaiting_payment = False
    
    async def handle_card_details(self, transcribed_text):
        """Processes 16-digit card number input"""
        data_type = 1
        card_number = await self.extract_data(transcribed_text, data_type)
        if card_number:
            self.card_number = card_number  # Store the card number
            await self.send(json.dumps({"message": "Card number received. Now, please provide the card expiry date in MM/YY format."}))
            self.awaiting_card_number = False  # Reset after receiving card
            self.awaiting_expiry_date = True  # Move to next step
        else:
            await self.send(json.dumps({"error": "Invalid card number. Please provide a valid 16-digit number."}))
    
    async def handle_expiry_date(self, transcribed_text):
        """Processes card expiry date input"""
        data_type = 2
        print("DATA TYPE BEFORE EXTRACT DATA: ",data_type)
        expiry_date = await self.extract_data(transcribed_text, data_type)
        print("DATA TYPE AFTER EXTRACT DATA: ",data_type)
        if expiry_date:
            self.expiry_date = expiry_date  # Store the expiry date
            await self.send(json.dumps({"message": "Expiry date received. Now, please provide the CVV (3 or 4 digits)."}))
            self.awaiting_expiry_date = False
            self.awaiting_cvv = True  # Move to next step
        else:
            await self.send(json.dumps({"error": "Invalid expiry date. Please provide in MM/YY format."}))

    async def handle_cvv(self, transcribed_text):
        """Processes CVV input"""
        data_type = 3
        cvv = await self.extract_data(transcribed_text, data_type)
        if cvv:
            self.cvv = cvv  # Store the CVV
            CC_number_value = self.card_number.get("message")
            CC_expiry_value = self.expiry_date.get("message")
            cvv_value = self.cvv.get("message")
            # await self.send(json.dumps({"message": "CVV received. Confirming details and processing payment..."}))
            message = (f"You entered the following details: \n"
                f"Card Number: {CC_number_value}\n"
                f"Expiry Date: {CC_expiry_value}\n"
                f"CVV: {cvv_value}\n"
                "Is this correct? Please say 'yes' or 'no'.")
            await self.send(json.dumps({"message": message}))
            self.awaiting_cvv = False
            # self.awaiting_after_confirmation = True
            # self.awaiting_cvv = False
            # self.awaiting_card_confirmation = True
            self.awaiting_after_confirmation = True
            # await self.confirm_card_details()  # Call payment processing function
        else:
            self.awaiting_card_number = True
            await self.send(json.dumps({"error": "Invalid CVV. Please provide a valid 3-digit (or 4-digit for AMEX) number."}))



    async def process_account_number(self, transcribed_text):
        """Extracts patient account number"""
        data_type = 0
        self.patient_account_number = await self.extract_data(transcribed_text, data_type)
        
        if self.patient_account_number:
            await self.send(json.dumps({"message": f"Is this your account number: {self.patient_account_number}? Please say 'yes' or 'no'."}))
            self.awaiting_account_confirmation = True
        else:
            await self.send(json.dumps({"error": "No valid account number detected. Please try again."}))

    
    async def disconnect(self, close_code):
        print("WebSocket Disconnected.")
    
    async def confirm_card_details(self):
        """Asks the user to confirm the extracted card details."""
        message = (f"You entered the following details: \n"
                f"Card Number: {self.card_number[-4:].rjust(len(self.card_number), '*')}\n"
                f"Expiry Date: {self.expiry_date}\n"
                f"CVV: {'***' if len(self.cvv) == 3 else '****'}\n"
                "Is this correct? Please say 'yes' or 'no'.")
        await self.send(json.dumps({"message": message}))
        self.awaiting_cvv = False
        self.awaiting_after_confirmation = True
        # self.awaiting_card_confirmation = True

    async def handle_card_confirmation(self, transcribed_text):
        """Handles user's confirmation response for card details."""
        confirmation = self.process_confirmation(transcribed_text)
        if confirmation == "yes":
            await self.send(json.dumps({"message": "Card details confirmed. Processing payment..."}))
            self.awaiting_after_confirmation = False
            # await self.process_payment()
        elif confirmation == "no":
            await self.send(json.dumps({"message": "Card details incorrect. Please re-enter your card number."}))
            self.reset_card_details()
        else:
            await self.send(json.dumps({"error": "Invalid response. Please say 'yes' or 'no'."}))

    def verify_account_number(self, account_number):
        """Checks if the given patient account number exists in the database."""

        print(f"🔍 Checking account number: {account_number}")  

        # If account_number is a dict, extract the actual value
        if isinstance(account_number, dict):  
            account_number = account_number.get("message", "").strip()  # <-- Fix: Get from "message"

        print(f"✅ Extracted account number: {account_number}")  

        # Ensure it's a string before querying
        if not account_number.isdigit():
            print("❌ Invalid account number format!")
            return False

        conn = sqlite3.connect("/home/adam/adam/IVR/Patient_Payment_IVR/ivr_apis/patients.db")  
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM patients WHERE account_number = ?", (account_number,))
        result = cursor.fetchone()

        conn.close()
        return result is not None

    def transcribe_audio_from_base64(self, base64_audio):
        """
        Decodes a base64 audio string, writes it to a temporary file,
        and transcribes it using the whisper_model.
        Returns the transcription result.
        """
        # Decode the base64 audio data
        try:
            audio_data = base64.b64decode(base64_audio)
        except Exception as e:
            print("Error decoding base64 audio:", e)
            return None

        # Write audio data to a temporary file with an appropriate suffix (e.g., .webm or .mp3)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file.flush()
            tmp_file_path = tmp_file.name

        try:
            # Call whisper_model.transcribe on the temporary file
            result = whisper_model.transcribe(tmp_file_path, fp16=False, language="english")
        except Exception as e:
            print("Error during transcription:", e)
            result = None
        finally:
            # Optionally, delete the temporary file after processing
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                print("Error removing temporary file:", e)
        
        return result

    def fetch_due_amount(self, account_number):
        """Fetches the due amount for a given patient account number."""
        print(f"🔍 Checking account number for due amount: {account_number} (Type: {type(account_number)})")  

        # If account_number is a dict, extract the actual value
        if isinstance(account_number, dict):  
            account_number = account_number.get("message", "").strip()  # <-- Fix: Get from "message"

        print(f"✅ Extracted account number: {account_number}")  

        # Ensure it's a string before querying
        if not account_number.isdigit():
            print("❌ Invalid account number format!")
            return False
        
        conn = sqlite3.connect("/home/adam/adam/IVR/Patient_Payment_IVR/ivr_apis/patients.db")
        cursor = conn.cursor()

        cursor.execute("SELECT due_amount FROM patients WHERE account_number = ?", (account_number,))
        result = cursor.fetchone()

        conn.close()

        return result[0] if result else 0
    
    def extract_payment_amount(self, text):
        """Extracts the payment amount from the spoken text."""
        import re
        pattern = r"(\d+(\.\d{1,2})?)"  # Matches numbers like 50 or 50.75

        match = re.search(pattern, text)
        if match:
            return float(match.group(1))  # Convert to float
        return None


    # # async def receive(self, text_data=None, bytes_data=None):
    # #     """Handles incoming WebSocket messages"""
    # #     data = json.loads(text_data)

    # #     if "audio_chunk" in data and "type" in data:
    # #         transcribed_text = whisper_model.transcribe(data["audio_chunk"], fp16=False, language="english")["text"]
    # #         print("Transcribed text:", transcribed_text)
    # #         response = await self.extract_data(transcribed_text, data["type"])
    # #         await self.send(json.dumps({"transcribed_text": transcribed_text, "extracted_info": response}))
    def process_confirmation(self, text):
        """Determines if user said 'yes' or 'no'"""
        text = text.lower().strip()
        if "yes" in text:
            return "yes"
        elif "no" in text:
            return "no"
        return "unclear"
    
    # async def disconnect(self, close_code):
    #     print("WebSocket Disconnected.")
