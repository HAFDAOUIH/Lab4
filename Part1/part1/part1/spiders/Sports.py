import scrapy
from ..items import Part1Item

class TechnologySpider(scrapy.Spider):
    name = 'Sports'
    start_urls = ['https://www.akhbarona.com/sport/articles/']

    def parse(self, response):
        liens = response.css('div.short')
        for lien in liens:
            article_url = lien.css('div.image a::attr(href)').get()
            yield response.follow(article_url, callback=self.parse_content)

        next_page_links = response.css('div#box_pagination  a.page_groups::attr(href)').getall()

        for next_page_link in next_page_links:
            yield response.follow(next_page_link, callback=self.parse)

    def parse_content(self, response):
        item = Part1Item()
        item['Content'] = response.css('div#article_body div#bodystr').xpath('string()').get()
        yield item

