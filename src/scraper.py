import praw
import json
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID")
    client_secret=os.getenv("CLIENT_SECRET")
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def scrape_subreddit(subreddit_name, limit=1000):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=limit):
        post_data = {
            'id': post.id,
            'title': post.title,
            'text': post.selftext,
            'created_utc': post.created_utc,
            'socre': post.score,
            'num_comments': post.num_comments,
            'subreddit': subreddit_name
        }
        posts.append(post_data)
    return posts

suicidal_posts = scrape_subreddit("SuicideWatch", limit=1000)
normal_posts = scrape_subreddit("CasualConversation", limit=1000)
with open('data/raw/suicidal_posts.json', 'w') as f:
    json.dump(suicidal_posts, f)
with open("data/raw/normal_posts.json", 'w') as f:
    json.dump(normal_posts, f)