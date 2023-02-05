---
title: "deep_learning"
layout: archive
permalink: categories/category2
author_profile: true
sidebar_main: true
---

## 테스트

{% assign posts = site.categories.category2 %}
{% for post in posts %} {% include archive-single2.html type=page.entries_layout %} {% endfor %}