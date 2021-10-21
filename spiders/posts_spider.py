import scrapy


class ForumSpider(scrapy.Spider):
    name = "talkeczema"

    start_urls = [
        'https://www.talkhealthpartnership.com/talkeczema/forums'
    ]

    def parse(self,response):
        forum_links = response.css('h5>a')
        yield from response.follow_all(forum_links,self.parse_forum)

    def parse_forum(self, response):
        post_links = response.css('a.x-topictitle')[1:]
        yield from response.follow_all(post_links, self.parse_post)

        pagination_links = response.css('li.next a')
        yield from response.follow_all(pagination_links, self.parse_forum)


    def parse_post(self, response):
        for post in response.css('div.postbody'):
            yield {
                'title': post.css('h3 a::text').get(),
                'author': post.css('p.author a.username::text').get(),
                'date': post.css('p.author::text').getall()[-1].rstrip(),
                'text': ' '.join((x.strip().strip('\n') for x in post.css('div.content *::text').getall()))}

        next_page = response.css('li.next a::attr(href)').get()
        if next_page:

            yield response.follow(next_page, callback=self.parse_post)
