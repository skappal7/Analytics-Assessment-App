mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\ "\n\
" > ~/ .streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enablesCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
