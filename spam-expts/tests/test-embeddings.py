{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create embeddings on the spam dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using embedded DuckDB without persistence: data will be transient\n",
      "No embedding_function provided, using default embedding function: SentenceTransformerEmbeddingFunction\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Default embedding function: <chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction object at 0x7f326c1fd730>\n"
     ]
    }
   ],
   "source": [
    "import chromadb\n",
    "from chromadb.config import Settings\n",
    "from chromadb.utils import embedding_functions\n",
    "\n",
    "sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=\"all-distilroberta-v1\")\n",
    "\n",
    "print(f'Default embedding function: {sentence_transformer_ef}')\n",
    "\n",
    "client = chromadb.Client(Settings(persist_directory=\"./db\")) # Does persistence work??\n",
    "\n",
    "\n",
    "collection = client.create_collection(name=\"spam-dataset\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "student_info = []\n",
    "student_info.append(\"\"\"\n",
    "Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,\n",
    "is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking\n",
    "in her free time in hopes of working at a tech company after graduating from the University of Washington.\n",
    "\"\"\")\n",
    "\n",
    "student_info.append(\"\"\"\n",
    "The university chess club provides an outlet for students to come together and enjoy playing\n",
    "the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning\n",
    "the rules to experienced tournament players. The club typically meets a few times per week to play casual games,\n",
    "participate in tournaments, analyze famous chess matches, and improve members' skills.\n",
    "\"\"\")\n",
    "\n",
    "student_info.append(\"\"\"\n",
    "The University of Washington, founded in 1861 in Seattle, is a public research university\n",
    "with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.\n",
    "As the flagship institution of the six public universities in Washington state,\n",
    "UW encompasses over 500 buildings and 20 million square feet of space,\n",
    "including one of the largest library systems in the world.\n",
    "\"\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No embedding_function provided, using default embedding function: SentenceTransformerEmbeddingFunction\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " count: 3\n",
      "List my collection:\n",
      " [Collection(name=span-dataset)]\n",
      "name='span-dataset' metadata=None\n",
      "Results: \n",
      " {'ids': [['id3', 'id2']], 'embeddings': None, 'documents': [['\\nThe University of Washington, founded in 1861 in Seattle, is a public research university\\nwith over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.\\nAs the flagship institution of the six public universities in Washington state,\\nUW encompasses over 500 buildings and 20 million square feet of space,\\nincluding one of the largest library systems in the world.\\n', \"\\nThe university chess club provides an outlet for students to come together and enjoy playing\\nthe classic strategy game of chess. Members of all skill levels are welcome, from beginners learning\\nthe rules to experienced tournament players. The club typically meets a few times per week to play casual games,\\nparticipate in tournaments, analyze famous chess matches, and improve members' skills.\\n\"]], 'metadatas': [[{'source': 'university info'}, {'source': 'club info'}]], 'distances': [[1.2249188750985962e-13, 1.5811126232147217]]}\n",
      "--------------------\n",
      "['\\nThe University of Washington, founded in 1861 in Seattle, is a public research university\\nwith over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.\\nAs the flagship institution of the six public universities in Washington state,\\nUW encompasses over 500 buildings and 20 million square feet of space,\\nincluding one of the largest library systems in the world.\\n', \"\\nThe university chess club provides an outlet for students to come together and enjoy playing\\nthe classic strategy game of chess. Members of all skill levels are welcome, from beginners learning\\nthe rules to experienced tournament players. The club typically meets a few times per week to play casual games,\\nparticipate in tournaments, analyze famous chess matches, and improve members' skills.\\n\"]\n"
     ]
    }
   ],
   "source": [
    "collection.add(\n",
    "    documents = [info for info in student_info],\n",
    "    metadatas = [{\"source\": \"student info\"},{\"source\": \"club info\"},{'source':'university info'}],\n",
    "    ids = [\"id1\", \"id2\", \"id3\"]\n",
    ")\n",
    "\n",
    "print(f' count: {collection.count()}')\n",
    "\n",
    "\n",
    "print(f'List my collection:\\n {client.list_collections()}')\n",
    "print(collection)\n",
    "\n",
    "results = collection.query(\n",
    "    query_texts=[\"What is the student name?\"],\n",
    "    n_results=2\n",
    ")\n",
    "\n",
    "\n",
    "results = collection.query(\n",
    "    query_texts=[\"What pastimes are available to students?\"],\n",
    "    n_results=1\n",
    ")\n",
    "\n",
    "\n",
    "results = collection.query(\n",
    "    query_texts=[student_info[2]],\n",
    "    n_results=2\n",
    ")\n",
    "\n",
    "print(f'Results: \\n {results}')\n",
    "\n",
    "print(f'{20 * \"-\"}')\n",
    "for doc in results['documents']:\n",
    "    print(doc)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "babyagi",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
