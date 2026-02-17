import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import DataTable from './DataTable';

interface TestItem {
  id: string;
  name: string;
  value: number;
}

const columns = [
  { key: 'name', header: 'Name', render: (item: TestItem) => item.name },
  { key: 'value', header: 'Value', render: (item: TestItem) => `$${item.value}` },
];

const data: TestItem[] = [
  { id: '1', name: 'Apple', value: 150 },
  { id: '2', name: 'Google', value: 2800 },
];

describe('DataTable', () => {
  it('renders column headers', () => {
    render(<DataTable columns={columns} data={data} keyExtractor={(i) => i.id} />);
    expect(screen.getByText('Name')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();
  });

  it('renders row data', () => {
    render(<DataTable columns={columns} data={data} keyExtractor={(i) => i.id} />);
    expect(screen.getByText('Apple')).toBeInTheDocument();
    expect(screen.getByText('$150')).toBeInTheDocument();
    expect(screen.getByText('Google')).toBeInTheDocument();
    expect(screen.getByText('$2800')).toBeInTheDocument();
  });

  it('shows empty message when no data', () => {
    render(<DataTable columns={columns} data={[]} keyExtractor={(i: TestItem) => i.id} emptyMessage="No items" />);
    expect(screen.getByText('No items')).toBeInTheDocument();
  });

  it('uses default empty message', () => {
    render(<DataTable columns={columns} data={[]} keyExtractor={(i: TestItem) => i.id} />);
    expect(screen.getByText('No data')).toBeInTheDocument();
  });
});
