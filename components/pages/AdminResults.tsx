import React, { useEffect, useState } from 'react';

type Clip = {
    id: number;
    cashier_name: string;
    from_time: string;
    to_time: string;
    file_path: string;
    created_at: string;
};

const AdminResults: React.FC = () => {
    const [clips, setClips] = useState<Clip[]>([]);
    const [cashier, setCashier] = useState('');
    const [from, setFrom] = useState('');
    const [to, setTo] = useState('');
    const [token, setToken] = useState<string | null>(null);

    useEffect(() => {
        const saved = localStorage.getItem('auth_token');
        if (saved) setToken(saved);
    }, []);

    const fetchClips = async () => {
        const params = new URLSearchParams();
        if (cashier) params.append('cashier', cashier);
        if (from) params.append('from', from);
        if (to) params.append('to', to);
        const res = await fetch('/api/clips?' + params.toString(), {
            headers: { Authorization: `Bearer ${token}` }
        });
        if (!res.ok) return;
        const data = await res.json();
        setClips(data);
    };

    useEffect(() => { if (token) fetchClips(); }, [token]);

    return (
        <div className="flex flex-col gap-4">
            <div className="p-4 bg-gray-800/60 rounded flex flex-wrap gap-2 items-end">
                <label className="text-sm">Cashier
                    <input className="ml-2 px-2 py-1 rounded bg-gray-700" value={cashier} onChange={e => setCashier(e.target.value)} />
                </label>
                <label className="text-sm">From
                    <input type="datetime-local" className="ml-2 px-2 py-1 rounded bg-gray-700" value={from} onChange={e => setFrom(e.target.value)} />
                </label>
                <label className="text-sm">To
                    <input type="datetime-local" className="ml-2 px-2 py-1 rounded bg-gray-700" value={to} onChange={e => setTo(e.target.value)} />
                </label>
                <button className="px-4 py-2 rounded bg-cyan-600" onClick={fetchClips}>Filter</button>
            </div>

            <div className="overflow-auto rounded border border-gray-700">
                <table className="min-w-full text-sm">
                    <thead className="bg-gray-800">
                        <tr>
                            <th className="p-2 text-left">Cashier</th>
                            <th className="p-2 text-left">From</th>
                            <th className="p-2 text-left">To</th>
                            <th className="p-2 text-left">Created</th>
                            <th className="p-2">Video</th>
                        </tr>
                    </thead>
                    <tbody>
                        {clips.map(c => (
                            <tr key={c.id} className="border-t border-gray-700">
                                <td className="p-2">{c.cashier_name}</td>
                                <td className="p-2">{new Date(c.from_time).toLocaleString()}</td>
                                <td className="p-2">{new Date(c.to_time).toLocaleString()}</td>
                                <td className="p-2">{new Date(c.created_at).toLocaleString()}</td>
                                <td className="p-2">
                                    <video src={c.file_path} controls className="w-64" />
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default AdminResults;


