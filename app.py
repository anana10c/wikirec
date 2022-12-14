import queue
from collections import defaultdict
import urllib

import numpy as np
import wikipedia
import wikipedia2vec
from wikipedia2vec import Wikipedia2Vec
from wikimapper import WikiMapper

from flask import Flask, request, render_template, session
from flask_session import Session

app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

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
	session["current_id"] = root_id

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

			try:
				feedback_key = tuple(sorted([root_id, wikidata_id]))
				feedback_weight = 1.0 + 0.1 * session["feedback"][feedback_key]
			except:
				feedback_weight = 1.0
		except KeyError:
			# print(f"KeyError on page score for {title}")
			wikidata_id = None
			page_score = 0
			feedback_weight = 1.0

		return (feedback_weight * (content_score + page_score), title, 2 if feedback_weight > 1 else int(content_score > page_score * 20), wikidata_id)

	checked = set(root_id)
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
	for _, root_link, _, _ in on_page_results:
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


def fetch_display_results(results):
	if results is None:
		return None, "Sorry! There was an error in retrieving the page."
	if len(results) == 0:
		return None, "Sorry! We couldn't find any results."
	data = []
	for _, title, explanation, _ in results:
		try:
			page = wikipedia.page(title, auto_suggest=False)
			summ = page.summary
			if len(summ) > 600:
				summ = summ[:600] + "..."
			url = page.url
		except:
			continue
		if explanation == 2:
			explanation_str = "This page was previously marked as a good recommendation through feedback."
		elif explanation == 1:
			explanation_str = "The content of this page is particularly relevant."
		else:
			assert explanation == 0
			explanation_str = "This page has a relatively high importance."
		data.append((pretty_title(title), url, summ, explanation_str))
	return data, None


def record_feedback(feedback_form):
	if "current_id" not in session or session["current_id"] is None:
		return "Sorry! Due to an error in retrieving the page ID, feedback could not be recorded."
	
	for i in range(len(session["results"])):
		result_key = "result" + str(i)
		if result_key in feedback_form:
			wikidata_id = session["results"][i][3]
			if wikidata_id is not None:
				feedback_key = tuple(sorted([session["current_id"], wikidata_id]))
				session["feedback"][feedback_key] += 1 if feedback_form[result_key] == "like" else -2
	return "Feedback submitted!"


# if __name__ == "__main__":
# 	load_models()
# 	link = "https://en.wikipedia.org/wiki/Human%E2%80%93computer_interaction"
# 	# link = "https://en.wikipedia.org/wiki/M%C4%81ori_language"
# 	print(compute_results(link, 20, True))


@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "POST":
		if "search" in request.form:
			link = request.form["wikilink"]
			num = request.form["numResults"]
			include_on_page = "checkPageLink" in request.form

			session["link"] = link

			results = compute_results(link, int(num), include_on_page)
			session["results"] = results
			display_results, message = fetch_display_results(results)
			return render_template("index.html", results=display_results, message=message, link=link)
		elif "feedback" in request.form:
			message = record_feedback(request.form)
			display_results, message2 = fetch_display_results(session["results"])
			return render_template("index.html", results=display_results, message=message if message2 is None else message2, link=session["link"])
		elif "clear" in request.form:
			session["feedback"].clear()
			message = "Feedback cleared!"
			return render_template("index.html", results=None, message=message, link=None)
		else:
			raise NotImplementedError

	return render_template("index.html", results=None, message=None, link=None)


@app.before_first_request
def setup():
    load_models()
    session["results"] = None
    session["feedback"] = defaultdict(int)


if __name__ == "__main__":
	app.run()
