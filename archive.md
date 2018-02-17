---
layout: page
title: Archive
---

{% for category in site.categories %}
<h3>{{ category | first }} （{{ category | last | size }} 篇）</h3>
<!-- <small>（{{ category | last | size }} 篇）</small> -->
<ul>
    {% for post in category.last %}
        <li>
          <a href="{{ post.url }}">{{ post.title }}</a>&nbsp;&nbsp;<small>{{ post.date | date_to_string }}</small>
        </li>
    {% endfor %}
</ul>
{% endfor %}
