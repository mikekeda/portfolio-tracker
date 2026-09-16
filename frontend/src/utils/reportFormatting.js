const escapeHtml = (value) => value
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;')
  .replace(/'/g, '&#39;');

export const markdownToHtml = (markdown) => {
  if (!markdown) return '';
  const bold = (s) => s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  const lines = String(markdown).split('\n').map(escapeHtml);
  const output = [];
  let inList = false;

  for (const line of lines) {
    if (line.startsWith('### ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h3>${bold(line.slice(4))}</h3>`);
    } else if (line.startsWith('## ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h2>${bold(line.slice(3))}</h2>`);
    } else if (line.startsWith('# ')) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<h1>${bold(line.slice(2))}</h1>`);
    } else if (/^[\s]*[-*] /.test(line)) {
      const content = line.replace(/^[\s]*[-*] /, '');
      if (!inList) { output.push('<ul>'); inList = true; }
      output.push(`<li>${bold(content)}</li>`);
    } else if (line.trim()) {
      if (inList) { output.push('</ul>'); inList = false; }
      output.push(`<p>${bold(line)}</p>`);
    } else {
      if (inList) { output.push('</ul>'); inList = false; }
    }
  }
  if (inList) output.push('</ul>');

  return output.join('\n');
};

// Guidance keeps the source unit; legacy reports must not silently become USD.
export const formatGuidance = (value, unit, revenue = false) => {
  if (!Number.isFinite(value)) return '—';
  const amount = revenue ? `${(value / 1000).toFixed(2)}B` : value.toFixed(2);
  if (!unit) return `${amount} (${revenue ? 'currency' : 'unit'} not recorded)`;
  if (!revenue && (unit === 'GBp' || unit === 'GBX')) return `${amount}p`;
  if (!revenue && unit === 'USc') return `${amount} US¢`;
  // ISO codes avoid ambiguous dollar symbols and are escaped by React.
  return `${unit} ${amount}`;
};
