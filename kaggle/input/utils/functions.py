"""In this Python file can be found all the functions used to tackle the LLM Science Kaggle Challenge divided by the respective sections."""
##############################################################################################################################
# Libraries
##############################################################################################################################
import ctypes

libc = ctypes.CDLL("libc.so.6")

import gc
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from faiss import read_index, index_factory
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from tqdm.notebook import tqdm
import blingfire as bf
from collections import Counter
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)


##############################################################################################################################
# Functions to calculate the metric specified into the Kaggle challenge
##############################################################################################################################
# source: https://www.kaggle.com/code/nandeshwar/mean-average-precision-map-k-metric-explained-code
def apk(actual: list, predicted: list, k: int = 10):
    """
    Compute the average precision at k.

    This function computes the average prescision at k between two lists of items.
    ----------
    Parameters
    ----------
    - actual: list
    A list of elements that are to be predicted (order doesn't matter)
    - predicted: list
    A list of predicted elements (order does matter)
    - k: int, optional
    The maximum number of predicted elements
    -------
    Returns
    -------
    - score: double
    The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual: list, predicted: list, k: int = 10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists of lists of items.
    ----------
    Parameters
    ----------
    - actual: list
    A list of lists of elements that are to be predicted (order doesn't matter in the lists)
    - predicted: list
    A list of lists of predicted elements (order matters in the lists)
    - k: int, optional
    The maximum number of predicted elements
    -------
    Returns
    -------
    - score: double
    The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


##############################################################################################################################
# Functions to import a LLM model from Hugging Face
##############################################################################################################################
def load_model(model_name: str, bnb_config, max_memory_val: int = 24e3):
    """
    Loads the specific model from the Hugging Face hub, together with its tokenizer.
    It uses GPU if available and sets its maximum memory capacity to max_memory_val.
    ----------
    Parameters
    ----------
    - model_name: string
    The name of the LLM model path to use from Hugging Face
    - bnb_config: object
    Bits and bytes configuration that quantises the model to 4 bits
    - max_memory_val: integer
    Represents the maximum GPU(s) memory capacity
    -------
    Returns
    -------
    - model: object
    The LLM model specified by the model_name parameter
    - tokenizer: object
    The LLM respective tokenizer
    """
    n_gpus = torch.cuda.device_count()
    max_memory = f"{max_memory_val}MB"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # dispatch efficiently the model on the available ressources
        max_memory={i: max_memory for i in range(n_gpus)},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    # Needed for LLama tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    """
    Loads the model in a 4 bit format to optimise the computation and store memory required.
    -----------
    Parameters
    -----------
    -------
    Returns
    -------
    - bnb_config: object
    The bits and bytes configuration that quantises the model file to 4 bits
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    return bnb_config


##############################################################################################################################
# Functions to calculate the inference for the specific LLM model
##############################################################################################################################
def recursive_inference(
    system_prompt,
    human_prompt,
    tokenizer,
    device,
    model,
    counter=0,
    previous_answer_list=[],
    max_repetitions=1
):
    """
    Get the three most likely options given by the model.

    When the model gives less than 3 options, it does another iteration.
    The maximum number of iterations is 5.
    This function conducts recursive inference to obtain likely answers from a language model. Here's how it works:
    Conduct recursive inference to obtain likely answers from a language model.

    It takes multiple parameters, including the system prompt, human prompt, context, and optional parameters for controlling the recursive process.
    The function constructs an input text by combining the system prompt, human prompt, and context.
    It tokenizes the input text using the provided tokenizer.
    The model generates a response based on the tokenized input, and the function decodes and cleans the response.
    If the response does not meet the desired answer format (e.g., less than three options), the function may recursively call itself to generate additional responses.
    The recursive process continues until the desired answer format is achieved.
    The function returns the answer list and the counter, which tracks the number of iterations.

    Args:
        system_prompt (str): The system's prompt.
        human_prompt (str): The human's prompt.
        context (str): Additional context text.
        counter (int, optional): Counter for recursive iterations. Default is 0.
        previous_answer_list (list, optional): Previous answer list. Default is [].
        max_repetitions (int, optional): Maximum recursive iterations. Default is 1.
        tokenizer: The model's tokenizer.
        device: The device for model execution.
        model: The question-answering model.

    Returns:
        tuple: A tuple containing the answer list and the counter.
    """
    # Model inference per sample
    # Specify input
    text = system_prompt + human_prompt
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # Get answer
    # Adjust max_new_tokens variable to 6 (maximum number of tokens the model can generate to answer the input)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode output & print it
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Some text cleaning
    answer_list = (
        res.replace("</s>", " ")
        .split(":>>")[-1]
        .strip()
        .split("\n")[0]
        .strip()
        .split(".")[0]
        .strip()
        .split(",")
    )
    # Call the model recursively max_repetitions times until it gives the correct answer format
    if len(answer_list) < 3 and counter < max_repetitions:
        answer_list, counter = recursive_inference(
            system_prompt,
            human_prompt,
            counter + 1,
            previous_answer_list=answer_list,
            max_repetitions=max_repetitions,
            model=model,
            device=device
        )
        if len(previous_answer_list) > len(answer_list):
            return previous_answer_list, counter
        else:
            return answer_list, counter
    else:
        return answer_list, counter


def recursive_inference_1(
    tokenizer, device, model, system_prompt: str, human_prompt: str, counter: int = 0, max_repetitions: int = 5
):
    """
    Gets the three most likely options given by the model.
    When the model gives less than 3 options, it does another iteration.
    The maximum number of iterations is 5.
    -----------
    Parameters
    ----------
    - system_prompt: string
    Message to indicate the LLM model in which type of language it has to answer and in which format
    - human_prompt: string
    Message that contains the context (if available) and the respective multi-choice question in this case
    - counter: int
    Indicates the number of times this function has been used until it gets the correct format
    - max_repetitions: int
    Maximum number of time this function can be recursively called, even if the correct format is still not given.
    --------
    Returns
    -------
    - res: string
    The whole answer that de LLM model gives and where the system_prompt and human_prompt parameters are concatenated
    - answer_list: list
    The list of the options that for the model are the ones that answer more accurately to the question
    - counter:
    Depicts the same as the parameter with the same name
    """
    # Model inference per sample
    # Specify input
    text = system_prompt + human_prompt
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # Get answer
    # Adjust max_new_tokens variable to 6 (maximum number of tokens the model can generate to answer the input)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"],
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode output & print it
    res = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Some text cleaning
    answer_list = (
        res.replace("</s>", " ")
        .split(":>>")[-1]
        .strip()
        .split("\n")[0]
        .strip()
        .split(".")[0]
        .strip()
        .split(",")
    )

    # Call the model recursively max_repetitions times until it gives the correct answer format
    if len(answer_list) < 3 and counter < max_repetitions:
        counter += 1
        res, answer_list, counter = recursive_inference_1(
            tokenizer, device, model, system_prompt, human_prompt, counter, max_repetitions = max_repetitions
        )
    else:
        return res, answer_list, counter

    return res, answer_list, counter


def get_most_likely_answers(tokenizer, device, model, system_prompt, human_prompt, repetitions = 10):
    """
    Uses the model to get the correct answer and iterates over the same multichoice question [reptitions] times.
    The answer will be a string with the top 3 most repeated answer letters given by the model.
    Uses the provided language model to generate answers to a given prompt and extracts the most likely answers based on repetitions.
    Args:
        tokenizer (object): The tokenizer for the language model.
        device (str): The device to perform inference on (e.g., "cuda" or "cpu").
        model (object): The language model for generating answers.
        system_prompt (str): The system-provided prompt text.
        human_prompt (str): The user-provided (human) prompt text.
        repetitions (int, optional): The number of times to repeat the generation process. Default is 10.

    Returns:
        list: A list of generated answers for each repetition.
        list: The top three most frequently occurring answers.
    """
    # Model inference per sample
    # Specify input
    text = system_prompt + human_prompt
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    # Initialise list for the model answers appending
    answers = []
    
    for i in range(repetitions):
        # Get answer
        # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
        outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=6, pad_token_id=tokenizer.eos_token_id)
        # Decode output & print it
        output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res = output.replace('</s>', ' ').split(':>>')[-1].strip().split('\n')[0].strip().split('.')[0].strip().split(',')
    
        # Save the answers in a list
        answers.extend(res)

    # Count the values into the results list and sort them by their frequency 
    sorted_counter = sorted(Counter(answers).items(), key = lambda x: x[1], reverse = True)
    
    # Create a list with just the most frequent top 3 values
    answer_list = [i[0] for i in sorted_counter]
    if len(answer_list) > 3:
        answer_list = answer_list[:3]
        
    return res, answer_list


def create_correct_incorrect_questions(tokenizer, device, model, system_prompt, data, index):
    """
    Uses the model to get whether an option from a sample is correct or not.
    The inner loop iterates 5 times per question, corresponding to each option.
    The outer loop iterates 3 times to get the top 3 options with more correct outputs by the model.
    Args:
        tokenizer (object): The tokenizer for the language model.
        device (str): The device to perform inference on (e.g., "cuda" or "cpu").
        model (object): The language model for generating questions and answers.
        system_prompt (str): The system-provided prompt text.
        data (pandas.DataFrame): A DataFrame containing the prompts and options.
        index (int): An index or identifier used for constructing human prompts.

    Returns:
        list: A list of most frequently occurring correct options.
    """
    # Model inference per sample
    options_list = ['A', 'B', 'C', 'D', 'E']
    # Initialise list for the sample answers appending
    answers_sample = []
    ## Outer loop
    for _ in range(3):
        ## Inner loop
        for opt in options_list:
            # Specify input
            text = system_prompt + human_prompt_one_option_answer(data, opt, index)
            # Tokenize input text
            inputs = tokenizer(text, return_tensors="pt").to(device)
            # Get answer
            # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
            outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"], max_new_tokens=8, pad_token_id=tokenizer.eos_token_id)
            # Decode output & print it
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            res = output.replace('</s>', ' ').split(':>>')[-1].strip().split('\n')[0].strip().split('.')[0].strip().split(',')
    
            # Save the correct sample answers in a list
            if 'correct' in res:
                answers_sample.extend(opt)
            
    # Count the values into the results list and sort them by their frequency 
    sorted_counter = sorted(Counter(answers_sample).items(), key = lambda x: x[1], reverse = True)
    
    # Create a list with just the most frequent top 3 values
    answer_list = [i[0] for i in sorted_counter]
    if len(answer_list) > 3:
        answer_list = answer_list[:3]
        
    return answer_list


def human_prompt(data_to_test, n: int = 0):
    """
    Creates a single string in which the question and the 5 options are set together with the context.
    Each section is divided by a given tag as it was stablishe in the finetuning process (<<:>>).
    ----------
    Parameters
    ----------
    - n: integer
    The index number of the correspondent sample into the dataset
    -------
    Returns
    -------
    - final_prompt: string
    The final message with the corresponding tags (<<:>>) to be introduced to the model
    """
    message = data_to_test.iloc[n]

    context = message["context"]
    question = message["question"]
    options = message["options"]

    template = "<<Context:>>\n{context}\n\n<<Question:>>\n{question}\n\n<<Options:>>\n{options}\n\n<<Assistant:>>"

    temp_prompt = PromptTemplate(
        template=template, input_variables=["context", "question", "options"]
    )
    final_prompt = temp_prompt.format(
        context=context, question=question, options=options
    )

    return final_prompt


def human_prompt_without_context(data_to_test, n: int = 0):
    """
    Generate a human-like prompt without context for testing with given data.
    Args:
        data_to_test (DataFrame): A DataFrame containing test data, with 'question' and 'options' columns.
        n (int, optional): Index to select a specific row from the DataFrame. Default is 0.

    Returns:
        str: A formatted prompt for a question with options.
    """
    message = data_to_test.iloc[n]

    question = message['question']
    options = message['options']

    template = "<<Question:>>\n{question}\n\n<<Options:>>\n{options}\n\n<<Assistant:>>"

    temp_prompt = PromptTemplate(template = template, input_variables = ['question', 'options'])
    final_prompt = temp_prompt.format(question = question, options = options)

    return final_prompt


def human_prompt_one_option_answer(data, opt: int, n: int = 0):
    """
    Generate a prompt for a single-choice question and answer task.
    This function takes a DataFrame containing prompts and their associated options, and generates a formatted prompt
    with a specific question and one option.
    Parameters:
        data (pandas.DataFrame): A DataFrame containing the prompts and options.
        opt (int): The index of the option to be presented in the prompt.
        n (int, optional): The index of the message in the DataFrame. Default is 0.

    Returns:
        str: A formatted prompt containing the specified question and option.
    """
    message = data.iloc[n]

    question = message['prompt']
    option = message[opt]

    template = "<<Question:>>\n{question}\n\n<<Option:>>\n{option}\n\n<<Assistant:>>"

    temp_prompt = PromptTemplate(template = template, input_variables = ['question', 'option'])
    final_prompt = temp_prompt.format(question = question, option = option)

    return final_prompt


##############################################################################################################################
# Functions to preprocess the source Wikipedia file to get the contexts
##############################################################################################################################
def process_documents(
    documents: Iterable[str],
    document_ids: Iterable,
    split_sentences: bool = True,
    filter_len: int = 3,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.
    ----------
    Parameters
    ----------
    - documents: iterable
    Contains documents which are strings
    - document_ids: iterable
    Contains document unique identifiers
    - document_type: string
    Denotes the document type to be processed
    - document_sections: list
    The list of sections for a given document type to process
    - split_sentences: boolean
    Flag to determine whether to further split sections into sentences
    - filter_len: integer
    Minimum character length of a sentence (otherwise filter out)
    - disable_progress_bar: boolean
    Flag to disable tqdm progress bar
    -------
    Returns
    -------
    - df: object
    Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(
            df.text.values,
            df.document_id.values,
            df.offset.values,
            filter_len,
            disable_progress_bar,
        )
    return df


def sectionize_documents(
    documents: Iterable[str], document_ids: Iterable, disable_progress_bar: bool = False
) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).
    ----------
    Parameters
    ----------
    - documents: iterable
    Contains documents which are strings
    - document_ids: iterable
    Containsdocument unique identifiers
    - disable_progress_bar: boolan
    Flag to disable tqdm progress bar
    -------
    Returns
    -------
    - _df: object
    Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(
        zip(document_ids, documents), total=len(documents), disable=disable_progress_bar
    ):
        row = {}
        text, start, end = (document, 0, len(document))
        row["document_id"] = document_id
        row["text"] = text
        row["offset"] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(["document_id", "offset"]).reset_index(drop=True)
    else:
        return _df


def sentencize(
    documents: Iterable[str],
    document_ids: Iterable,
    offsets: Iterable[tuple[int, int]],
    filter_len: int = 3,
    disable_progress_bar: bool = False,
) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents` to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the location in the original documents.
    ----------
    Parameters
    ----------
    - documents: iterable
    Contains documents which are strings
    - document_ids: iterable
    Contains document unique identifiers
    - offsets: iterable
    Tuple of the start and end indices
    - filter_len: integer
    Minimum character length of a sentence (otherwise filter out)
    -------
    Returns
    -------
    - df: object
    Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    document_sentences = []
    for document, document_id, offset in tqdm(
        zip(documents, document_ids, offsets),
        total=len(documents),
        disable=disable_progress_bar,
    ):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1] - o[0] > filter_len:
                    sentence = document[o[0] : o[1]]
                    abs_offsets = (o[0] + offset[0], o[1] + offset[0])
                    row = {}
                    row["document_id"] = document_id
                    row["text"] = sentence
                    row["offset"] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    df = pd.DataFrame(document_sentences)

    return df


def split_lists(mylist, chunk_size):
    """
    Splits a list based on chunk size.
    
    Args:
    mylist: list to be splitted.
    chunk_size: Size the the splitted lists. 
    Returns: List of lists. 
    """
    return [
        mylist[offs : offs + chunk_size] for offs in range(0, len(mylist), chunk_size)
    ]


def retrieval(df_valid, modified_texts, stop_words):
    """
    Perform retrieval of relevant articles from a given dataset.

    This function is responsible for performing the retrieval of relevant articles. Here's how it works:

    It takes three main inputs: df_valid, modified_texts, and an optional list of stop_words. df_valid is the input DataFrame with queries, and modified_texts is a list of preprocessed texts. stop_words is an optional parameter that allows you to specify words to be excluded from the analysis.
    The function preprocesses the text data from the input DataFrame df_valid. It concatenates multiple fields from the DataFrame and tokenizes them. This is done to prepare the text data for the retrieval process.
    The function then creates TF-IDF vectors for both the input text data and the documents. The TF-IDF (Term Frequency-Inverse Document Frequency) score is a measure of the importance of words in a document relative to a collection of documents.
    The function calculates the relevance scores for the documents using these TF-IDF vectors and the cosine similarity metric. Cosine similarity measures the cosine of the angle between two non-zero vectors.
    It retrieves the top relevant documents based on the calculated relevance scores and returns a list of indices and relevance scores for these documents.

    Args:
        df_valid (DataFrame): The input DataFrame with queries.
        modified_texts (list): Preprocessed text data.
        stop_words (list, optional): List of stop words. Default is None.

    Returns:
        tuple: A tuple containing indices and scores of retrieved articles.
    """
    corpus_df_valid = df_valid.apply(
        lambda row: f'{row["prompt"]}\n{row["prompt"]}\n{row["prompt"]}\n{row["A"]}\n{row["B"]}\n{row["C"]}\n{row["D"]}\n{row["E"]}',
        axis=1,
    ).values
    vectorizer1 = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
    )
    vectorizer1.fit(corpus_df_valid)
    vocab_df_valid = vectorizer1.get_feature_names_out()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[\w/.-]+\b|!|/|\?|\"|\'",
        stop_words=stop_words,
        vocabulary=vocab_df_valid,
    )
    vectorizer.fit(modified_texts[:500000])
    corpus_tf_idf = vectorizer.transform(corpus_df_valid)

    print(f"length of vectorizer vocab is {len(vectorizer.get_feature_names_out())}")

    chunk_size = 100000
    top_per_chunk = 10
    top_per_query = 10

    all_chunk_top_indices = []
    all_chunk_top_values = []

    for idx in tqdm(range(0, len(modified_texts), chunk_size)):
        wiki_vectors = vectorizer.transform(modified_texts[idx : idx + chunk_size])
        # [total_vocab:total_docs]*[total_vocab:chunk_size].T
        temp_scores = (corpus_tf_idf * wiki_vectors.T).toarray()
        chunk_top_indices = temp_scores.argpartition(-top_per_chunk, axis=1)[
            :, -top_per_chunk:
        ]
        chunk_top_values = temp_scores[
            np.arange(temp_scores.shape[0])[:, np.newaxis], chunk_top_indices
        ]

        all_chunk_top_indices.append(chunk_top_indices + idx)
        all_chunk_top_values.append(chunk_top_values)

    top_indices_array = np.concatenate(all_chunk_top_indices, axis=1)
    top_values_array = np.concatenate(all_chunk_top_values, axis=1)

    merged_top_scores = np.sort(top_values_array, axis=1)[:, -top_per_query:]
    merged_top_indices = top_values_array.argsort(axis=1)[:, -top_per_query:]
    articles_indices = top_indices_array[
        np.arange(top_indices_array.shape[0])[:, np.newaxis], merged_top_indices
    ]

    return articles_indices, merged_top_scores


##############################################################################################################################
# Functions to get the context for the multi-choice questions
##############################################################################################################################
def get_contexts():
    """
    The function in a nutshell gets the most appropiate context for each multi-choice question.
    The pipeline to achieve the previous functionality starts with embedding vector calculation of the Wikipedia documents with a length of
    MAX_LENGTH and throughout a sentence transformer model.
    With these embeddings, for the vector similarity calculation, the libary FAISS from Meta is required ans utilised.
    At the end, the document with the highest similarity is chosed and is merged with slices of the corresponding Wikipedia page it belongs.
    This is the context saved together with the respective question and options in the final dataset.
    ----------
    Parameters
    ----------
    -------
    Returns
    -------
    - df_return: object
    Pandas DataFrame tha contains the columns `id`, `prompt`, `context`, `A`, `B`, `C`, `D`, `E`
    """
    # Some configuration constants for the paths and the transformer model
    SIM_MODEL = "./kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2"
    DEVICE = 0
    MAX_LENGTH = 384
    BATCH_SIZE = 16

    WIKI_PATH = "./kaggle/input/wikipedia-20230701"

    trn = pd.read_csv("./kaggle/input/kaggle-llm-science-exam/test.csv").drop(
        "id", axis=1
    )

    model = SentenceTransformer(SIM_MODEL, device="cuda")
    model.max_seq_length = MAX_LENGTH
    model = model.half()

    sentence_index = read_index(
        "./kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index"
    )

    prompt_embeddings = model.encode(
        trn.apply(
            lambda row: f"{row['prompt']}\n{row['A']}\n{row['B']}\n{row['C']}\n{row['D']}\n{row['E']}",
            axis=1,
        ).values,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
    _ = gc.collect()

    # Get the top 20 pages that are likely to contain the topic of interest
    search_score, search_index = sentence_index.search(prompt_embeddings, 20)

    # Save memory - delete sentence_index since it is no longer necessary
    del sentence_index
    del prompt_embeddings
    _ = gc.collect()
    libc.malloc_trim(0)

    df = pd.read_parquet(
        "./kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
        columns=["id", "file"],
    )

    # Get the article and associated file location using the index
    wikipedia_file_data = []

    for i, (scr, idx) in tqdm(
        enumerate(zip(search_score, search_index)), total=len(search_score)
    ):
        scr_idx = idx
        _df = df.loc[scr_idx].copy()
        _df["prompt_id"] = i
        wikipedia_file_data.append(_df)
    wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
    wikipedia_file_data = (
        wikipedia_file_data[["id", "prompt_id", "file"]]
        .drop_duplicates()
        .sort_values(["file", "id"])
        .reset_index(drop=True)
    )

    # Save memory - delete df since it is no longer necessary
    del df
    _ = gc.collect()
    libc.malloc_trim(0)

    # Get the full text data
    wiki_text_data = []

    for file in tqdm(
        wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())
    ):
        _id = [
            str(i)
            for i in wikipedia_file_data[wikipedia_file_data["file"] == file][
                "id"
            ].tolist()
        ]
        _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=["id", "text", "title"])

        _df_temp = _df[_df["id"].isin(_id)].copy()
        del _df
        _ = gc.collect()
        libc.malloc_trim(0)
        wiki_text_data.append(_df_temp)
    wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
    _ = gc.collect()

    # Parse documents into sentences
    processed_wiki_text_data = process_documents(
        wiki_text_data.text.values, wiki_text_data.id.values
    )

    # Get embeddings of the wiki text data
    wiki_data_embeddings = model.encode(
        processed_wiki_text_data.text,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True)  # .half()
    
    wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()

    _ = gc.collect()

    # Combine all answers
    trn["answer_all"] = trn.apply(
        lambda x: " ".join([x["A"], x["B"], x["C"], x["D"], x["E"]]), axis=1
    )

    # Search using the prompt and answers to guide the search
    trn["prompt_answer_stem"] = trn["prompt"] + " " + trn["answer_all"]

    question_embeddings = model.encode(
        trn.prompt_answer_stem.values,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    question_embeddings = question_embeddings.detach().cpu().numpy()

    # Parameter to determine how many relevant sentences to include
    NUM_SENTENCES_INCLUDE = 6

    # List containing just Context
    contexts = []

    for r in tqdm(trn.itertuples(), total=len(trn)):
        prompt_id = r.Index

        prompt_indices = processed_wiki_text_data[
            processed_wiki_text_data["document_id"].isin(
                wikipedia_file_data[wikipedia_file_data["prompt_id"] == prompt_id][
                    "id"
                ].values
            )
        ].index.values

        if prompt_indices.shape[0] > 0:
            ## WHERE DOES FAISS COME FROM?
            prompt_index = index_factory(wiki_data_embeddings.shape[1], "Flat")
            prompt_index.add(wiki_data_embeddings[prompt_indices])

            context = ""

            # Get the top matches
            ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
            for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
                context += (
                    processed_wiki_text_data.loc[prompt_indices]["text"].iloc[_i] + " "
                )
        contexts.append(context)

    trn["context"] = contexts

    df_return = trn[
        ["prompt", "context", "A", "B", "C", "D", "E"]
    ]  # .to_csv("./test_context.csv", index=False)

    return df_return


def get_relevant_documents_parsed(df_valid, stop_words):
    """
    Retrieve relevant documents from a parsed dataset of paraphrases.

    This function is designed to retrieve relevant documents from a dataset of parsed paraphrases. Here's how it works:

    It takes an input DataFrame df_valid as its argument, which contains queries or prompts.
    The function first loads a pre-processed dataset of paraphrases using the load_from_disk function. This dataset likely contains titles, sections, and text data for various paraphrased documents.
    It preprocesses the text data from the dataset, concatenating the title, section, and text while replacing newline characters and single quotes with spaces.
    The input DataFrame df_valid is divided into smaller chunks to efficiently process a large number of queries. This is done in a for loop, iterating through chunks of the DataFrame.
    For each chunk, it calls the retrieval function to retrieve relevant articles from the chunk based on their relevance to the queries in the DataFrame. The retrieved articles' indices and relevance scores are stored.
    Finally, the function assembles the retrieved articles, sorts them by relevance, and returns them as a list.

    Args:
        df_valid (DataFrame): The input DataFrame with queries.

    Returns:
        list: A list of retrieved articles.
    """

    df_chunk_size = 600
    paraphs_parsed_dataset = load_from_disk(
        "./kaggle/working/all-paraphs-parsed-expanded"
    )
    modified_texts = paraphs_parsed_dataset.map(
        lambda example: {
            "temp_text": f"{example['title']} {example['section']} {example['text']}".replace(
                "\n", " "
            ).replace(
                "'", ""
            )
        },
        num_proc=2,
    )["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx : idx + df_chunk_size]

        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts, stop_words)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (
            articles_values_array[index],
            paraphs_parsed_dataset[idx.item()]["title"],
            paraphs_parsed_dataset[idx.item()]["text"],
        )
        for index, idx in enumerate(article_indices_array.reshape(-1))
    ]
    retrieved_articles = split_lists(articles_flatten, top_per_query)
    return retrieved_articles


def get_relevant_documents(df_valid, stop_words):
    """
    Retrieve relevant documents from a different dataset of preprocessed documents.

    This function is similar to the previous one, but it is designed to work with a different dataset of preprocessed documents. Here's how it works:

    It takes the input DataFrame df_valid, which contains queries or prompts.
    Like the previous function, it loads a pre-processed dataset using load_from_disk. However, this dataset may contain different types of documents.
    It preprocesses the text data from the dataset in a similar manner as the previous function, removing unwanted characters and normalizing text.
    The input DataFrame is divided into smaller chunks for efficient processing.
    For each chunk, it calls the retrieval function to retrieve relevant articles from the chunk based on their relevance to the queries. The retrieved articles' indices and relevance scores are stored.
    Finally, the function assembles the retrieved articles, sorts them by relevance, and returns them as a list.
    Args:
        df_valid (DataFrame): The input DataFrame with queries.

    Returns:
        list: A list of retrieved articles.
    """
    df_chunk_size = 800

    cohere_dataset_filtered = load_from_disk("./kaggle/working/stem-wiki-cohere-no-emb")
    modified_texts = cohere_dataset_filtered.map(
        lambda example: {
            "temp_text": unicodedata.normalize(
                "NFKD", f"{example['title']} {example['text']}"
            ).replace('"', "")
        },
        num_proc=2,
    )["temp_text"]

    all_articles_indices = []
    all_articles_values = []
    for idx in tqdm(range(0, df_valid.shape[0], df_chunk_size)):
        df_valid_ = df_valid.iloc[idx: idx + df_chunk_size]

        articles_indices, merged_top_scores = retrieval(df_valid_, modified_texts, stop_words)
        all_articles_indices.append(articles_indices)
        all_articles_values.append(merged_top_scores)

    article_indices_array = np.concatenate(all_articles_indices, axis=0)
    articles_values_array = np.concatenate(all_articles_values, axis=0).reshape(-1)

    top_per_query = article_indices_array.shape[1]
    articles_flatten = [
        (
            articles_values_array[index],
            cohere_dataset_filtered[idx.item()]["title"],
            unicodedata.normalize("NFKD", cohere_dataset_filtered[idx.item()]["text"]),
        )
        for index, idx in enumerate(article_indices_array.reshape(-1))
    ]
    retrieved_articles = split_lists(articles_flatten, top_per_query)
    return retrieved_articles


##############################################################################################################################
# Functions for finetuning the LLM model
##############################################################################################################################
def random_answer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate random answers for multiple-choice questions.

    This function generates random answers for multiple-choice questions. Here's how it works:

    It takes a DataFrame (df) that contains questions and available answer options.
    The function iterates through the rows of the DataFrame and checks if any of the answer options are missing (indicated by "-").
    For questions with missing options, the function randomly assigns available answer options to fill the gaps, ensuring that each question has a unique set of answer options.
    The modified DataFrame is returned as the output, now with all questions having complete answer options.

    Args:
        df (DataFrame): Input DataFrame with questions and options.

    Returns:
        DataFrame: The modified DataFrame with random answer assignments.
    """
    full_options = ["A", "B", "C", "D", "E"]
    copy_df = df.copy()
    for idx, id, p1, p2, p3 in df.itertuples():
        if p1 == "-":
            random_idx_1 = np.random.randint(len(full_options))
            random_option_1 = full_options[random_idx_1]
            del full_options[random_idx_1]

            random_idx_2 = np.random.randint(len(full_options))
            random_option_2 = full_options[random_idx_2]
            del full_options[random_idx_2]

            random_idx_3 = np.random.randint(len(full_options))
            random_option_3 = full_options[random_idx_3]
            del full_options[random_idx_3]

            copy_df.loc[idx, "prediction1"] = random_option_1
            copy_df.loc[idx, "prediction2"] = random_option_2
            copy_df.loc[idx, "prediction3"] = random_option_3
        elif p2 == "-":
            actual_options = list(set(full_options).difference(set([p1])))

            random_idx_2 = np.random.randint(len(actual_options))
            random_option_2 = actual_options[random_idx_2]
            del actual_options[random_idx_2]

            random_idx_3 = np.random.randint(len(actual_options))
            random_option_3 = actual_options[random_idx_3]
            del actual_options[random_idx_3]

            copy_df.loc[idx, "prediction2"] = random_option_2
            copy_df.loc[idx, "prediction3"] = random_option_3

        elif p3 == "-":
            actual_options = list(set(full_options).difference(set([p1, p2])))

            random_idx_3 = np.random.randint(len(actual_options))
            random_option_3 = actual_options[random_idx_3]
            del actual_options[random_idx_3]

            copy_df.loc[idx, "prediction3"] = random_option_3

    return copy_df