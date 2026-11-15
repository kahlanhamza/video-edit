"""GitHub API helper — reads token from data/secrets/github_token (never prints it)."""
import json, os, subprocess, sys

TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "secrets", "github_token")
REPO = "kahlanhamza/video-edit"

def _token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()

def api(method, path, payload=None):
    url = f"https://api.github.com/repos/{REPO}/{path}" if not path.startswith("/") else path
    cmd = ["curl", "-s", "-X", method, "-H", "Authorization: Bearer " + _token(),
           "-H", "Accept: application/vnd.github+json", url]
    if payload is not None:
        cmd += ["-d", json.dumps(payload)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(r.stdout) if r.stdout.strip() else {}

def create_issue(title, body, labels=None):
    return api("POST", "issues", {"title": title, "body": body,
                                  "labels": labels or []}).get("number")

def comment(number, body):
    return api("POST", f"issues/{number}/comments", {"body": body}).get("id")

def close(number, state_reason="completed"):
    api("PATCH", f"issues/{number}", {"state": "closed", "state_reason": state_reason})

def create_release(tag, title, body, files=None):
    rel = api("POST", "releases", {"tag_name": tag, "target_commitish": "main",
                                   "name": title, "body": body, "draft": False})
    if "id" not in rel:
        return rel
    for f in (files or []):
        upload_url = rel["upload_url"].split("{")[0] + f"?name={os.path.basename(f)}"
        subprocess.run(["curl", "-s", "-X", "POST",
                        "-H", "Authorization: Bearer " + _token(),
                        "-H", "Content-Type: application/octet-stream",
                        "--data-binary", "@" + f, upload_url],
                       capture_output=True, text=True)
    return rel

def unauth_url(tag, asset):
    return f"https://github.com/{REPO}/releases/download/{tag}/{asset}"

if __name__ == "__main__":
    if sys.argv[1] == "new":
        print(create_issue(sys.argv[2], sys.stdin.read()))
    elif sys.argv[1] == "say":
        comment(int(sys.argv[2]), sys.stdin.read())
    elif sys.argv[1] == "close":
        close(int(sys.argv[2]))
