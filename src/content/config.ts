import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    pubDate: z.date(),
    abstract: z.string().optional(),// 文章摘要
  }),
});

export const collections = {
  posts,
};
