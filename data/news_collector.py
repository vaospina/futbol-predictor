"""
Recolector de noticias deportivas via RSS feeds.
"""
import feedparser
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

RSS_FEEDS = [
    {"url": "https://www.espn.com/espn/rss/soccer/news", "source": "ESPN"},
    {"url": "https://www.marca.com/rss/futbol.xml", "source": "Marca"},
    {"url": "https://www.bbc.co.uk/sport/football/rss.xml", "source": "BBC"},
    {"url": "https://www.goal.com/feeds/en/news", "source": "Goal.com"},
    {"url": "https://www.transfermarkt.com/rss/news", "source": "Transfermarkt"},
]


def fetch_news(team_names: list = None, max_per_feed: int = 20) -> list:
    """
    Obtiene noticias de todos los RSS feeds.
    Si team_names se proporciona, filtra por menciones del equipo.
    """
    all_news = []

    for feed_config in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_config["url"])
            entries = feed.entries[:max_per_feed]

            for entry in entries:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                published = entry.get("published_parsed")

                pub_date = None
                if published:
                    try:
                        pub_date = datetime(*published[:6])
                    except Exception:
                        pass

                news_item = {
                    "headline": title,
                    "summary": summary,
                    "source": feed_config["source"],
                    "published_at": pub_date,
                }

                if team_names:
                    text = f"{title} {summary}".lower()
                    for team in team_names:
                        if team.lower() in text:
                            news_item["matched_team"] = team
                            all_news.append(news_item)
                            break
                else:
                    all_news.append(news_item)

        except Exception as e:
            logger.warning(f"Error parseando RSS {feed_config['source']}: {e}")

    logger.info(f"Obtenidas {len(all_news)} noticias relevantes")
    return all_news


def fetch_team_news(team_name: str) -> list:
    return fetch_news(team_names=[team_name])
