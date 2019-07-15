'''
The web provides us with more data than any of us can read and understand, so we often want to work with that information programmatically
in order to make sense of it. Sometimes, that data is provided to us by website creators via . csv or comma-separated values files, or 
through an API (Application Programming Interface). Other times, we need to collect text from the web ourselves. 
This assignment will go over how to work with the Reauests and Beautiful  Soup Python packages in order to make use of data from web pages.
The Requests module lets you integrate your Python programs with web services, while the Beautiful Soup module is designed to make 
screen-scraping get done quickly. 
'''

from urllib import request
from bs4 import BeautifulSoup

page = request.urlopen('https://assets.digitalocean.com/articles/eng_python/beautiful-soup/mockturtle.html')
print(page)
# <http.client.HTTPResponse object at 0x000001EA25674A58>

soup = BeautifulSoup(page,'html.parser')
print(soup.prettify)

print(soup.title)
# <title>Turtle Soup</title>

print(soup.title.name)
# title

print(soup.get_text())
# Printed the complete text without any html tags

# Finding all <p> tags
print(soup.find_all('p'))

# Printing first <p> tag's content
print(soup.find_all('p')[0].get_text())

# Displaying content of all <p> tags
for i in range(len(soup.find_all('p'))):
    print(str(i) +' Paragraph : \n' , soup.find_all('p')[i].get_text())
    print('\n')
    
'''
Output:

0 Paragraph : 
 Beautiful Soup, so rich and green,
  Waiting in a hot tureen!
  Who for such dainties would not stoop?
  Soup of the evening, beautiful Soup!
  Soup of the evening, beautiful Soup!


1 Paragraph : 
 Beau--ootiful Soo--oop!
  Beau--ootiful Soo--oop!
  Soo--oop of the e--e--evening,
  Beautiful, beautiful Soup!


2 Paragraph : 
 Beautiful Soup! Who cares for fish,
  Game or any other dish?
  Who would not give all else for two
  Pennyworth only of Beautiful Soup?
  Pennyworth only of beautiful Soup?


3 Paragraph : 
 Beau--ootiful Soo--oop!
  Beau--ootiful Soo--oop!
  Soo--oop of the e--e--evening,
  Beautiful, beauti--FUL SOUP!
  
'''


br_tag = soup.find_all('br')
print(br_tag)

print(len(br_tag))
# 18

print(soup.find_all('a')) # No anchor tag thats why it returned empty list

verse = soup.find_all(class_ = 'verse')
print(verse)

chorus = soup.find_all(class_ = 'chorus')
print(chorus)

# Displaying content of <id> tags

first = soup.find_all(id='first')
print(first)

third = soup.find_all(id='third')
print(third)

four = soup.find_all(id='fourth')
print(four)

print(first[0].get_text())

print(four[0].get_text())

a = soup.find(class_ = 'verse' , id='first')
print(a)

b = soup.find_all(class_ = 'verse' , id='first')
print(b)
