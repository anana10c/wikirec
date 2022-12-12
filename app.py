import queue
import numpy as np
import urllib
import wikipedia
import wikipedia2vec
from wikipedia2vec import Wikipedia2Vec
from wikimapper import WikiMapper

from flask import Flask, request, render_template

MODEL_FILE = "D:/wikirec_storage/enwiki_20180420_100d.pkl"
PAGERANK_FILE = "D:/wikirec_storage/2022-11-10.allwiki.links.rank"
MAPPER_FILE = "D:/wikirec_storage/index_enwiki-latest.db"


def load_models():
	global wiki2vec
	wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

	print("wikipedia2vec loaded!")

	global pagerank_dict
	pagerank_dict = {}
	with open(PAGERANK_FILE, "r") as f:
		for line in f:
			pageid, rank = line.split()
			pagerank_dict[pageid] = float(rank)

	print("pagerank loaded!")

	global mapper
	mapper = WikiMapper(MAPPER_FILE)

	print("mapper loaded!")

	global like_feedback
	like_feedback = set()
	global dislike_feedback
	dislike_feedback = set()


def extract_title(link):
	idx = link.rfind('/')
	return urllib.parse.unquote(link[idx+1:])


def pretty_title(title):
	return ' '.join(title.split('_'))


def compute_results(link, num=5, include_on_page=False):
	root_title = extract_title(link)
	print(f"getting results for {root_title}")

	try:
		root_page = wikipedia.page(root_title)
	except:
		return None

	try:
		root_vec = wiki2vec.get_entity_vector(pretty_title(root_title))
		root_vec = root_vec / np.linalg.norm(root_vec)
	except KeyError:
		return []
	root_id = mapper.title_to_id(root_title)

	def compute_rank(title):
		try:
			vec = wiki2vec.get_entity_vector(pretty_title(title))
			content_score = np.dot(root_vec, vec) / np.linalg.norm(vec)
		except KeyError:
			# print(f"KeyError on content score for {title}")
			content_score = 0

		try:
			wikidata_id = mapper.title_to_id(title)
			page_score = np.log10(pagerank_dict[wikidata_id]) / 50
		except KeyError:
			# print(f"KeyError on page score for {title}")
			page_score = 0

		return (content_score + page_score, title, content_score > page_score * 20)

	checked = set(mapper.title_to_id(root_title))
	on_page_results = queue.PriorityQueue(maxsize=20)
	for link in root_page.links:
		if mapper.title_to_id(link) not in checked:
			rank = compute_rank(link)
			if on_page_results.full():
				lowest = on_page_results.get()
				on_page_results.put(max(lowest, rank))
			else:
				on_page_results.put(rank)
			checked.add(mapper.title_to_id(link))
	on_page_results = on_page_results.queue

	results = queue.PriorityQueue(maxsize=num)
	for _, root_link, _ in on_page_results:
		try:
			secondary_page = wikipedia.page(root_link)
		except:
			continue
		for secondary_link in secondary_page.links:
			if mapper.title_to_id(secondary_link) not in checked:
				rank = compute_rank(secondary_link)
				if results.full():
					lowest = results.get()
					results.put(max(lowest, rank))
				else:
					results.put(rank)
				checked.add(mapper.title_to_id(secondary_link))

	if include_on_page:
		return list(reversed(sorted(results.queue + on_page_results)))[:min(num, len(results.queue) + len(on_page_results))]
	else:
		return list(reversed(sorted(results.queue)))[:min(num, len(results.queue))]


def fetch_results(results):
	if results is None:
		return None, "Sorry! There was an error in retrieving the page."
	if len(results) == 0:
		return None, "Sorry! We couldn't seem to find any results."
	data = []
	for _, title, explanation in results:
		try:
			page = wikipedia.page(title, auto_suggest=False)
			summ = page.summary
			if len(summ) > 600:
				summ = summ[:600] + "..."
			url = page.url
		except:
			continue
		explanation_str = "The content of this page is particularly relevant." if explanation else "This page has a relatively high importance."
		data.append((pretty_title(title), url, summ, explanation_str))
	return data, None


# if __name__ == "__main__":
# 	load_models()
# 	link = "https://en.wikipedia.org/wiki/Human%E2%80%93computer_interaction"
# 	# link = "https://en.wikipedia.org/wiki/M%C4%81ori_language"
# 	print(compute_results(link, 20, True))

app = Flask(__name__)

load_models()


@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		print(request.form)
		link = request.form["wikilink"]
		num = request.form["numResults"]
		include_on_page = "checkPageLink" in request.form
		results, message = fetch_results(compute_results(link, int(num), include_on_page))
		# return f"POST successful with link {link}"
		return render_template("index.html", results=results, message=message)
	else:
		return render_template("index.html", results=None, message=None)
