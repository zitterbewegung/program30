from haystack.utils import clean_wiki_text, convert_files_to_dicts, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
# SQLite Document Store
from haystack.document_stores import SQLDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline

document_store = SQLDocumentStore(url="sqlite:///qa.db")


def etl(question):
	# Let's first get some documents that we want to query
	# Here: 517 Wikipedia articles for Game of Thrones
	doc_dir = "data/article_txt_got"
	s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
	fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

	# convert files to dicts containing documents that can be indexed to our datastore
	# You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
	# It must take a str as input, and return a str.
	dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

	# We now have a list of dictionaries that we can write to our document store.
	# If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
	# The default format here is: {"name": "<some-document-name>, "text": "<the-actual-text>"}

	# Let's have a look at the first 3 entries:
	#print(dicts[:3])
	# Now, let's write the docs to our DB.
	document_store.write_documents(dicts)
	reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
	pipe = ExtractiveQAPipeline(reader, retriever)
	prediction = pipe.run(query="Who is the father of Arya Stark?", params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})                                                                              
	return print_answers(prediction, details="minimum")
