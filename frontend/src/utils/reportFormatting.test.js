import { markdownToHtml, formatGuidance, selectGuidance, hasGuidance } from './reportFormatting';

test('report HTML stays inert in paragraphs, headings, lists and bold text', () => {
  const payload = '# <img src=x onerror="alert(1)">\n## <svg onload="alert(1)">\n### <script>alert(1)</script>\n- **<iframe srcdoc="bad">**\n<img src=x> &lt;img src=x&gt;';
  const container = document.createElement('div');
  container.innerHTML = markdownToHtml(payload);
  expect(container.querySelector('img, svg, script, iframe')).toBeNull();
  expect(container.querySelector('h1').textContent).toContain('<img');
  expect(container.querySelector('li strong').textContent).toContain('<iframe');
  expect(container.textContent).toContain('&lt;img src=x&gt;');
});

test('ordinary report formatting and ampersands remain readable', () => {
  const container = document.createElement('div');
  container.innerHTML = markdownToHtml('## Results\n**Growth** & cash\n- One\n- Two');
  expect(container.querySelector('h2').textContent).toBe('Results');
  expect(container.querySelector('strong').textContent).toBe('Growth');
  expect(container.querySelectorAll('li')).toHaveLength(2);
  expect(container.textContent).toContain('Growth & cash');
  expect(markdownToHtml(null)).toBe('');
});

test('guidance uses reported units, including pence and zero', () => {
  expect(formatGuidance(44000, 'EUR', true)).toBe('EUR 44.00B');
  expect(formatGuidance(178.5, 'GBp')).toBe('178.50p');
  expect(formatGuidance(0, 'USD')).toBe('USD 0.00');
  expect(formatGuidance(-1.25, 'USD')).toBe('USD -1.25');
});

test('legacy guidance never acquires an invented dollar currency', () => {
  expect(formatGuidance(44000, null, true)).toBe('44.00B (currency not recorded)');
  expect(formatGuidance(178.5, undefined)).toBe('178.50 (unit not recorded)');
  expect(formatGuidance(null, 'USD')).toBe('—');
});

test('unit-only SEC guidance cannot suppress same-day PR guidance', () => {
  const pr = { revenue_guidance: { next_year: 44000, currency: 'EUR' } };
  expect(selectGuidance({ guidance: { eps_guidance: { unit: 'USD' }, revenue_guidance: { currency: 'EUR' } }, pr_guidance: pr }))
    .toEqual({ guidance: pr, guidanceFromPR: true });
  const zero = { eps_guidance: { next_year: 0, unit: 'USD' } };
  expect(selectGuidance({ guidance: zero, pr_guidance: pr })).toEqual({ guidance: zero, guidanceFromPR: false });
});

test('blank outlook is empty; margin-only and qualitative guidance are meaningful', () => {
  expect(hasGuidance({ outlook_commentary: '  ', revenue_guidance: { currency: 'GBP' } })).toBe(false);
  expect(hasGuidance({ operating_margin_guidance: 0 })).toBe(true);
  expect(hasGuidance({ outlook_commentary: 'Demand expected to improve' })).toBe(true);
});
