
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop redis 
==> Downloading Homebrew API data
✔︎ JSON API packages.arm64_tahoe.jws.json            Downloaded   15.5MB/ 15.5MB
Stopping `redis`... (might take a while)
==> Successfully stopped `redis` (label: sh.brew.redis)
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop mongodb-community@8.0
Stopping `mongodb-community@8.0`... (might take a while)
==> Successfully stopped `mongodb-community@8.0` (label: homebrew.mxcl.mongodb-c
gautamkoshta@Gautams-MacBook-Air ~ % brew services stop neo4j
Stopping `neo4j`... (might take a while)
==> Successfully stopped `neo4j` (label: homebrew.mxcl.neo4j)
gautamkoshta@Gautams-MacBook-Air ~ % 


brew services start redis 
brew services start mongodb-community@8.0
brew services start neo4j

brew services stop redis 
brew services stop mongodb-community@8.0
brew services stop neo4j